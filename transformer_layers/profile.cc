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
// Direct guest operations avoid the process creation required by the shell
// fallback. The library and headers must come from the selected gem5 build.
#include <gem5/m5ops.h>
#endif

namespace {

#if CFG_GEM5_PROFILE_REGIONS
const char* kGem5ProfileIndexPath = "gem5_profile_regions.tsv";

// Buffer index rows until finalization. Their allocations and string copies
// still execute between dumps; only index-file and index-console I/O is deferred.
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
        // The first stats operation probes for the shell helper and caches it.
        cached_m5_available =
            (std::system("command -v m5 >/dev/null 2>&1") == 0) ? 1 : 0;
    }
    return cached_m5_available == 1;
}
#endif

// A dump ends the current interval. Subsequent in-memory bookkeeping belongs
// to the next interval, or to post-ROI execution after the final dump.
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

    // Dump before appending the row. Reserve reduces vector growth, but strings
    // can allocate and this work remains inside the next measured interval.
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
    // Flush an earlier window before resetting counters. Callers must finalize
    // after their final dump; abrupt termination can lose buffered rows.
    finalizeTransformerStatsWindow();

    gem5_profile_dump_index = 0;

#if CFG_GEM5_PROFILE_REGIONS
    gem5_profile_scope = scope;
    gem5_profile_index_rows.clear();
    // The interleaved int8 paths close four regions per attention head before
    // the five block-level boundaries, so the row count scales with num_heads.
    // 64 covers BERT-base (12 heads) without reallocating; the vector grows if
    // a wider model needs more.
    gem5_profile_index_rows.reserve(64);
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
