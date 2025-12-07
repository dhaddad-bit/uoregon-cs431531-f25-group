#include <stdio.h>
#include <unistd.h>

#include "spmv.h"
#include "common.h"

static P_TYPE g_ticks_persecond = 0.0;

void InitTSC(void)
{
    uint64_t start_tick = ReadTSC();
    sleep(1);
    uint64_t end_tick = ReadTSC();

    g_ticks_persecond = (P_TYPE) (end_tick - start_tick);
    //fprintf(stderr, "%e ticks per second.\n", g_ticks_persecond);
}


P_TYPE ElapsedTime(uint64_t ticks)
{
    if (g_ticks_persecond == 0.0) {
        fprintf(stderr, "TSC timer has not been initialized.\n");
        return 0.0;
    }
    else {
        return (ticks / g_ticks_persecond);
    }
}
