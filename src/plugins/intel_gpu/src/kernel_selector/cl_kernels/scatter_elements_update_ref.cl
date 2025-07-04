// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define GET_INDICES_INDEX(idx_order) INPUT1_GET_INDEX(idx_order)
#define GET_UPDATES_INDEX(idx_order) INPUT2_GET_INDEX(idx_order)
#define GET_OUTPUT_INDEX(idx_order) OUTPUT_GET_INDEX(idx_order)
#define GET_INPUT_INDEX(idx_order) INPUT0_GET_INDEX(idx_order)

#if AXIS_VALUE == 0
    #define SIZE INPUT0_BATCH_NUM
    #define ASSIGN_INDEX(index) b = index
#elif AXIS_VALUE == 1
    #define SIZE INPUT0_FEATURE_NUM
    #define ASSIGN_INDEX(index) f = index
#endif
#if OUTPUT_DIMS == 4
    #define ORDER b,f,y,x
    #if AXIS_VALUE == 2
        #define SIZE INPUT0_SIZE_Y
        #define ASSIGN_INDEX(index) y = index
    #elif AXIS_VALUE == 3
        #define SIZE INPUT0_SIZE_X
        #define ASSIGN_INDEX(index) x = index
    #endif
#elif OUTPUT_DIMS == 5
    #define ORDER b,f,z,y,x
    #if AXIS_VALUE == 2
        #define SIZE INPUT0_SIZE_Z
        #define ASSIGN_INDEX(index) z = index
    #elif AXIS_VALUE == 3
        #define SIZE INPUT0_SIZE_Y
        #define ASSIGN_INDEX(index) y = index
    #elif AXIS_VALUE == 4
        #define SIZE INPUT0_SIZE_X
        #define ASSIGN_INDEX(index) x = index
    #endif
#elif OUTPUT_DIMS == 6
    #define ORDER b,f,w,z,y,x
    #if AXIS_VALUE == 2
        #define SIZE INPUT0_SIZE_W
        #define ASSIGN_INDEX(index) w = index
    #elif AXIS_VALUE == 3
        #define SIZE INPUT0_SIZE_Z
        #define ASSIGN_INDEX(index) z = index
    #elif AXIS_VALUE == 4
        #define SIZE INPUT0_SIZE_Y
        #define ASSIGN_INDEX(index) y = index
    #elif AXIS_VALUE == 5
        #define SIZE INPUT0_SIZE_X
        #define ASSIGN_INDEX(index) x = index
    #endif
#endif

#if OUTPUT_DIMS != INPUT2_DIMS
    #error "OUTPUT_DIMS is supposed to be same as INPUT2_DIMS"
#endif

#if ITER < 2 // First and second kernel only
    #ifdef REDUCE_MODE
        #define SUM_MODE 1
        #define PROD_MODE 2
        #define MIN_MODE 3
        #define MAX_MODE 4
        #define MEAN_MODE 5

        #if USE_INIT_VAL == 0
            #if REDUCE_MODE == SUM_MODE
                #define REDUCTION_NEUTRAL_VALUE INPUT1_VAL_ZERO
            #elif REDUCE_MODE == PROD_MODE
                #define REDUCTION_NEUTRAL_VALUE INPUT1_VAL_ONE
            #elif REDUCE_MODE == MIN_MODE
                #define REDUCTION_NEUTRAL_VALUE INPUT1_VAL_MAX
            #elif REDUCE_MODE == MAX_MODE
                #define REDUCTION_NEUTRAL_VALUE INPUT1_VAL_MIN
            #elif REDUCE_MODE == MEAN_MODE
                #define REDUCTION_NEUTRAL_VALUE INPUT0_VAL_ZERO
            #else
                #error "Invalid REDUCE_MODE value"
            #endif
        #endif

        inline int FUNC(to_fixed)(INPUT2_TYPE data_in)
        {
            float scaled = data_in;

            if (INPUT2_IS_FP && INPUT2_TYPE_SIZE == 2)
            {
                scaled = convert_float(scaled) * 65536.0f;
                scaled = clamp(scaled, -2147483648.0f, 2147483647.0f);
            }
            return convert_int_rte(scaled);
        }

        inline INPUT1_TYPE FUNC(count_add)(volatile __local INPUT1_TYPE *ptr, int val)
        {
            atomic_int *atomic_addr = ptr;
            return atomic_fetch_add(atomic_addr, val);
        }

        inline void FUNC(atomic_reduce)(volatile __local int *ptr, int val)
        {
            atomic_int *atomic_addr = ptr;
            #if REDUCE_MODE == PROD_MODE
                int old_val, new_val;
                do {
                    old_val = *ptr;
                    new_val = old_val * val;
                } while (atomic_cmpxchg(ptr, old_val, new_val) != old_val);
            #elif REDUCE_MODE == MIN_MODE
                atomic_fetch_min(atomic_addr, val);
            #elif REDUCE_MODE == MAX_MODE
                atomic_fetch_max(atomic_addr, val);
            #elif REDUCE_MODE == SUM_MODE
                atomic_fetch_add(atomic_addr, val);
            #elif REDUCE_MODE == MEAN_MODE
                atomic_fetch_add(atomic_addr, val);
            #else
                #error "Invalid REDUCE_MODE value"
            #endif
        }

        inline void FUNC(atomic_reduce_global)(volatile __global int *ptr, int val)
        {
            atomic_int *atomic_addr = ptr;
            #if REDUCE_MODE == PROD_MODE
                int old_val, new_val;
                do {
                    old_val = *ptr;
                    new_val = old_val * val;
                } while (atomic_cmpxchg(ptr, old_val, new_val) != old_val);
            #elif REDUCE_MODE == MIN_MODE
                atomic_fetch_min(atomic_addr, val);
            #elif REDUCE_MODE == MAX_MODE
                atomic_fetch_max(atomic_addr, val);
            #elif REDUCE_MODE == SUM_MODE
                atomic_fetch_add(atomic_addr, val);
            #elif REDUCE_MODE == MEAN_MODE
                atomic_fetch_add(atomic_addr, val);
            #else
                #error "Invalid REDUCE_MODE value"
            #endif
        }
    #endif
#endif

KERNEL(scatter_elements_update_ref)(OPTIONAL_SHAPE_INFO_ARG
                   const __global INPUT0_TYPE* data,
                   const __global INPUT1_TYPE* indices,
                   const __global INPUT2_TYPE* updates,
                   __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                   , FUSED_OPS_DECLS
#endif
#if REDUCE_MODE != 0
                   , __global INPUT1_TYPE* output_fp
#endif
#if ITER == 1
                   , __local INPUT1_TYPE* reduction_v
                   , __local INPUT1_TYPE* reduction_thread
#if REDUCE_MODE == MEAN_MODE
                   , __local INPUT1_TYPE* count_k
#endif
#endif
)
{
    const uint dim0 = get_global_id(0);
    const uint dim1 = get_global_id(1);
    const uint dim2 = get_global_id(2);
#if ITER == 0 // First kernel
    #if OUTPUT_DIMS == 4
        const uint x = dim0;
        const uint y = dim1;
        const uint f = dim2 % OUTPUT_FEATURE_NUM;
        const uint b = dim2 / OUTPUT_FEATURE_NUM;
    #elif OUTPUT_DIMS == 5
        const uint x = dim0 % OUTPUT_SIZE_X;
        const uint y = dim0 / OUTPUT_SIZE_X;
        const uint z = dim1;
        const uint f = dim2 % OUTPUT_FEATURE_NUM;
        const uint b = dim2 / OUTPUT_FEATURE_NUM;
    #elif OUTPUT_DIMS == 6
        const uint x = dim0 % OUTPUT_SIZE_X;
        const uint y = dim0 / OUTPUT_SIZE_X;
        const uint z = dim1 % OUTPUT_SIZE_Z;
        const uint w = dim1 / OUTPUT_SIZE_Z;
        const uint f = dim2 % OUTPUT_FEATURE_NUM;
        const uint b = dim2 / OUTPUT_FEATURE_NUM;
    #endif
    const uint input_idx = GET_INPUT_INDEX(ORDER);
    const uint output_idx = GET_OUTPUT_INDEX(ORDER);
    INPUT0_TYPE val = data[input_idx];

    #ifndef REDUCE_MODE
        #if HAS_FUSED_OPS
            FUSED_OPS_FIRST_KERNEL;
            output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT_FIRST_KERNEL);
        #else
            output[output_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
        #endif
    #else
        INPUT1_TYPE val_fp = FUNC_CALL(to_fixed)(val);

        #if HAS_FUSED_OPS
            FUSED_OPS_FIRST_KERNEL;
            output_fp[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT_FIRST_KERNEL);
        #else
            output_fp[output_idx] = ACTIVATION(val_fp, ACTIVATION_PARAMS);
        #endif

    #endif

#elif ITER == 1
    #ifdef REDUCE_MODE
        #if OUTPUT_DIMS == 4
            uint x = dim0 % INPUT2_SIZE_X;
            uint y = dim0 / INPUT2_SIZE_X;
            uint f = dim1 % INPUT2_FEATURE_NUM;
            uint b = dim2 % INPUT2_BATCH_NUM;
        #elif OUTPUT_DIMS == 5
            uint x = dim0 % INPUT2_SIZE_X;
            uint y = dim0 / INPUT2_SIZE_X;
            uint z = dim1 % INPUT2_SIZE_Z;
            uint f = dim1 / INPUT2_SIZE_Z;
            uint b = dim2 % INPUT2_BATCH_NUM;
        #elif OUTPUT_DIMS == 6
            uint x = dim0 % INPUT2_SIZE_X;
            uint y = dim0 / INPUT2_SIZE_X;
            uint z = dim1 % INPUT2_SIZE_Z;
            uint w = dim1 / INPUT2_SIZE_Z;
            uint f = dim2 % INPUT2_FEATURE_NUM;
            uint b = dim2 / INPUT2_FEATURE_NUM;
        #endif
        const uint indices_idx = GET_INDICES_INDEX(ORDER);
        const uint updates_idx = GET_UPDATES_INDEX(ORDER);
        INPUT2_TYPE val = updates[(int)updates_idx];
        INPUT1_TYPE index = indices[(int)indices_idx];

        if (index < 0) { index += SIZE; }
        ASSIGN_INDEX(index);
        const uint output_idx = GET_OUTPUT_INDEX(ORDER);

        reduction_thread[output_idx] = 0;
        #if USE_INIT_VAL == 0
            reduction_v[output_idx] = FUNC_CALL(to_fixed)(REDUCTION_NEUTRAL_VALUE);

            #if REDUCE_MODE == MEAN_MODE
                count_k[output_idx] = INPUT1_VAL_ZERO;
            #endif
        #endif
        barrier(CLK_LOCAL_MEM_FENCE);

        int val_fixed = FUNC_CALL(to_fixed)(val);

        FUNC_CALL(atomic_reduce)(&reduction_v[output_idx], val_fixed);
        INPUT1_TYPE write_thread = FUNC_CALL(count_add)(&reduction_thread[output_idx], 1);

        if (write_thread == 0) {
            #if USE_INIT_VAL == 0
                output_fp[output_idx] = FUNC_CALL(to_fixed)(REDUCTION_NEUTRAL_VALUE);
            #endif
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        #if REDUCE_MODE == MEAN_MODE
            FUNC_CALL(count_add)(&count_k[output_idx], INPUT1_VAL_ONE);
            reduction_v[output_idx] /= (count_k[output_idx] + USE_INIT_VAL);
        #endif
        if (write_thread == 0) {
            FUNC_CALL(atomic_reduce_global)(&output_fp[output_idx], reduction_v[output_idx]);
        }
    #else // REDUCE_MODE==NONE.
        uint ORDER;
        #if OUTPUT_DIMS == 4
            x = dim0;
            y = dim1;
            f = dim2 % INPUT2_FEATURE_NUM;
            b = dim2 / INPUT2_FEATURE_NUM;
        #elif OUTPUT_DIMS == 5
            x = dim0 % INPUT2_SIZE_X;
            y = dim0 / INPUT2_SIZE_X;
            z = dim1;
            f = dim2 % INPUT2_FEATURE_NUM;
            b = dim2 / INPUT2_FEATURE_NUM;
        #elif OUTPUT_DIMS == 6
            x = dim0 % INPUT2_SIZE_X;
            y = dim0 / INPUT2_SIZE_X;
            z = dim1 % INPUT2_SIZE_Z;
            w = dim1 / INPUT2_SIZE_Z;
            f = dim2 % INPUT2_FEATURE_NUM;
            b = dim2 / INPUT2_FEATURE_NUM;
        #endif
        const uint indices_idx = GET_INDICES_INDEX(ORDER);
        const uint updates_idx = GET_UPDATES_INDEX(ORDER);
        INPUT2_TYPE val = updates[(int)updates_idx];
        INPUT1_TYPE index = indices[(int)indices_idx];
        if (index < 0) {index += SIZE;}
        ASSIGN_INDEX(index);
        const uint output_idx = GET_OUTPUT_INDEX(ORDER);
        #if HAS_FUSED_OPS
            FUSED_OPS_SECOND_KERNEL;
            output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT_SECOND_KERNEL);
        #else
            output[output_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
        #endif
    #endif
#elif ITER == 2
    #if OUTPUT_DIMS == 4
        const uint x = dim0;
        const uint y = dim1;
        const uint f = dim2 % OUTPUT_FEATURE_NUM;
        const uint b = dim2 / OUTPUT_FEATURE_NUM;
    #elif OUTPUT_DIMS == 5
        const uint x = dim0 % OUTPUT_SIZE_X;
        const uint y = dim0 / OUTPUT_SIZE_X;
        const uint z = dim1;
        const uint f = dim2 % OUTPUT_FEATURE_NUM;
        const uint b = dim2 / OUTPUT_FEATURE_NUM;
    #elif OUTPUT_DIMS == 6
        const uint x = dim0 % OUTPUT_SIZE_X;
        const uint y = dim0 / OUTPUT_SIZE_X;
        const uint z = dim1 % OUTPUT_SIZE_Z;
        const uint w = dim1 / OUTPUT_SIZE_Z;
        const uint f = dim2 % OUTPUT_FEATURE_NUM;
        const uint b = dim2 / OUTPUT_FEATURE_NUM;
    #endif
    const uint input_idx = GET_INPUT_INDEX(ORDER);
    const uint output_idx = GET_OUTPUT_INDEX(ORDER);
    int val_fixed = output_fp[input_idx];
    float val_f32 = convert_float(val_fixed) / 65536.0f;
    INPUT2_TYPE val = TO_OUTPUT_TYPE(val_f32);

    #if HAS_FUSED_OPS
        FUSED_OPS_SECOND_KERNEL;
        output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT_SECOND_KERNEL);
    #else
        output[output_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
    #endif


#endif
}

#ifdef REDUCE_MODE
    #undef SUM_MODE
    #undef PROD_MODE
    #undef MIN_MODE
    #undef MAX_MODE
    #undef MEAN_MODE
    #undef REDUCTION_NEUTRAL_VALUE
#endif

#undef GET_INDICES_INDEX
#undef GET_UPDATES_INDEX
#undef GET_OUTPUT_INDEX
#undef ORDER
#undef SIZE
#undef ASSIGN_INDEX
