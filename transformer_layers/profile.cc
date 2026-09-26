#include "profile.h"

#include "run_mode_config.h"

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#if defined(I2CE_USE_LIBM5)
// Built against gem5's util/m5 (libm5.a) for the aarch64 guest. This issues the
// m5 op directly instead of forking a shell, which is the whole point of the
// option: std::system() inside a region boundary costs guest process creation
// plus exec, and that cost lands in the measured window.
#include <gem5/m5ops.h>
#endif

namespace {

#if CFG_GEM5_PROFILE_REGIONS
const char* kGem5ProfileIndexPath = "gem5_profile_regions.tsv";

// Region-index rows are accumulated here and written once, after the measured
// window closes.  Previously every checkpoint opened the file, appended a row
// and flushed std::cout, so file and console I/O were charged to the region
// that had just ended.
struct ProfileIndexRow {
    std::size_t dump_index;
    std::string checkpoint;
    std::string interval_since_previous;
};
std::vector<ProfileIndexRow> gem5_profile_index_rows;
std::string gem5_profile_scope;
bool gem5_profile_index_flushed = false;
#endif

std::size_t gem5_profile_dump_index = 0;

#if !defined(I2CE_USE_LIBM5)
bool m5Available() {
    static int cached_m5_available = -1;
    if (cached_m5_available < 0) {
        // Probed once, outside any steady-state region, and cached.
        cached_m5_available =
            (std::system("command -v m5 >/dev/null 2>&1") == 0) ? 1 : 0;
    }
    return cached_m5_available == 1;
}
#endif

// The only work that may happen on a region boundary: issue the stats op.
// Everything else is deferred so it cannot be attributed to the region.
void m5DumpStatsNow() {
#if defined(I2CE_USE_LIBM5)
    m5_dump_stats(0, 0);
#else
    if (m5Available()) {
        std::system("m5 dumpstats");
    }
#endif
}

void m5ResetStatsNow() {
#if defined(I2CE_USE_LIBM5)
    m5_reset_stats(0, 0);
#else
    if (m5Available()) {
        std::system("m5 resetstats");
    }
#endif
}

void m5DumpResetStatsNow() {
#if defined(I2CE_USE_LIBM5)
    m5_dump_reset_stats(0, 0);
#else
    if (m5Available()) {
        std::system("m5 dumpresetstats");
    }
#endif
}

#if CFG_GEM5_PROFILE_REGIONS
void dumpTransformerStatsCheckpoint(const char* checkpoint,
                                    const char* interval_since_previous) {
    gem5_profile_dump_index++;

    // Issue the stats dump first, then record the bookkeeping in memory.  The
    // push_back may allocate, but it happens after the boundary and is not file
    // or console I/O; reserve() below keeps it amortised.
    m5DumpStatsNow();

    gem5_profile_index_rows.push_back(
        ProfileIndexRow{gem5_profile_dump_index,
                        std::string(checkpoint),
                        std::string(interval_since_previous)});
}
#endif

} // namespace

void finalizeTransformerStatsWindow();

void resetTransformerStatsWindow(const char* scope) {
    // Flush whatever the previous window accumulated before starting a new one,
    // so an interrupted run still leaves the last completed window on disk. This
    // happens before the new window opens, so it is never measured.
    finalizeTransformerStatsWindow();

    gem5_profile_dump_index = 0;

#if CFG_GEM5_PROFILE_REGIONS
    gem5_profile_scope = scope;
    gem5_profile_index_rows.clear();
    // One transformer block produces six stage boundaries plus a final total.
    gem5_profile_index_rows.reserve(8);
    gem5_profile_index_flushed = false;
#else
    (void)scope;
#endif

    // Announce the scope before the window opens, never between boundaries.
    std::cout << "[GEM5_PROFILE] reset scope=" << scope << std::endl;
    m5ResetStatsNow();
}

void finalizeTransformerStatsWindow() {
#if CFG_GEM5_PROFILE_REGIONS
    if (gem5_profile_index_flushed) {
        return;
    }
    gem5_profile_index_flushed = true;

    std::ofstream index(kGem5ProfileIndexPath);
    if (index.is_open()) {
        index << "scope\t" << gem5_profile_scope << "\n";
        index << "dump_index\tcheckpoint\tinterval_since_previous\n";
        for (const ProfileIndexRow& row : gem5_profile_index_rows) {
            index << row.dump_index << "\t"
                  << row.checkpoint << "\t"
                  << row.interval_since_previous << "\n";
        }
    }

    for (const ProfileIndexRow& row : gem5_profile_index_rows) {
        std::cout << "[GEM5_PROFILE] dump " << row.dump_index
                  << " checkpoint=" << row.checkpoint
                  << " interval=" << row.interval_since_previous << "\n";
    }
    std::cout << std::flush;
#endif
}

void dumpTransformerStatsCheckpointIfProfiling(const char* checkpoint,
                                               const char* interval_since_previous) {
#if CFG_GEM5_PROFILE_REGIONS
    // Region profiling keeps cumulative snapshots.  Subtract adjacent dumps to
    // recover a per-region interval, while the last dump remains the total.
    dumpTransformerStatsCheckpoint(checkpoint, interval_since_previous);
#else
    (void)checkpoint;
    (void)interval_since_previous;
#endif
}

void dumpTransformerStatsLegacyBoundary(const char* checkpoint,
                                        const char* interval_since_previous) {
#if CFG_GEM5_PROFILE_REGIONS
    dumpTransformerStatsCheckpoint(checkpoint, interval_since_previous);
#else
    (void)checkpoint;
    (void)interval_since_previous;
    m5DumpResetStatsNow();
#endif
}
