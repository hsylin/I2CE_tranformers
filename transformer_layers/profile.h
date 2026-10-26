#pragma once

// Cumulative stats dumps delimit intervals, including any intervening setup,
// allocations, console output and in-memory profile bookkeeping. Finalization
// writes the buffered index after the caller's last dump; it does not issue a
// dump or reset itself. The caller owns the measurement boundaries.
//
// Define I2CE_USE_LIBM5 and link the matching guest libm5.a for direct operations.
// Otherwise each available m5 helper call forks a shell inside the guest.


void resetTransformerStatsWindow(const char* scope);

// Writes the buffered region index and echoes it. Call once, after the last
// boundary and outside the measured window. Idempotent.
void finalizeTransformerStatsWindow();

void dumpTransformerStatsCheckpointIfProfiling(const char* checkpoint,
                                               const char* interval_since_previous);

void dumpTransformerStatsLegacyBoundary(const char* checkpoint,
                                        const char* interval_since_previous);
