#pragma once

/*
 * Region profiling boundaries.
 *
 * The measured window must contain the work being measured and nothing else.
 * Region bookkeeping (the gem5_profile_regions.tsv index and its console echo)
 * is therefore buffered in memory and written by finalizeTransformerStatsWindow()
 * after the window closes, instead of opening a file and flushing std::cout on
 * every boundary.
 *
 * Build with -DI2CE_USE_LIBM5 and link gem5's libm5.a (util/m5, built for the
 * aarch64 guest) to issue the m5 ops directly. Without it the ops are invoked
 * through std::system("m5 ..."), which forks a shell inside the guest on every
 * boundary; that cost is charged to the measured window.
 */

void resetTransformerStatsWindow(const char* scope);

// Writes the buffered region index and echoes it. Call once, after the last
// boundary and outside the measured window. Idempotent.
void finalizeTransformerStatsWindow();

void dumpTransformerStatsCheckpointIfProfiling(const char* checkpoint,
                                               const char* interval_since_previous);

void dumpTransformerStatsLegacyBoundary(const char* checkpoint,
                                        const char* interval_since_previous);
