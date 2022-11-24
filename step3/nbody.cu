/**
 * @File nbody.cu
 *
 * Implementation of the N-Body problem
 *
 * Paralelní programování na GPU (PCG 2022)
 * Projekt c. 1 (cuda)
 * Login: xpolok03
 */

#include <cmath>
#include <cfloat>
#include "nbody.h"

/**
 * Aux function to get size of dynamic shared memory
 * Author: einpoklum  Source: https://stackoverflow.com/questions/42309369/can-my-kernel-code-tell-how-much-shared-memory-it-has-available
 * @return Size of dynamic shared memory
 */
__forceinline__ __device__ unsigned dynamic_smem_size() {
    unsigned ret;
    asm volatile ("mov.u32 %0, %dynamic_smem_size;" : "=r"(ret));
    return ret;
}

/**
 * CUDA kernel to calculate gravitation velocity
 * @param p       - particles
 * @param tmp_vel - temp array for velocities
 * @param N       - Number of particles
 * @param dt      - Size of the time step
 */
__global__ void calculate_velocity(t_particles p_curr,
                                   t_particles p_next,
                                   int N,
                                   float dt) {
    // All threads have to stay active, otherwise there won't thread to store values to shared memory
    extern __shared__ float shared[];
    unsigned global_id = threadIdx.x + blockIdx.x * blockDim.x;

    // Aux pointer for unified indexing and reinterpreting of memory
    auto *shared_pos = reinterpret_cast<float4 *>(shared);
    auto *shared_vel = reinterpret_cast<float3 *>( &shared[blockDim.x * POS_ELEMENTS] );

    // Auxiliary registers to store p1 and p2 current positions, weights and velocities
    float4 pos_p1 = p_curr.pos[global_id];
    float3 vel_p1 = p_curr.vel[global_id];
    float4 pos_p2;
    float3 vel_p2;

    // Aux variables used in loop below
    float ir, dx, dy, dz, ir3, Fg_dt_m2_r, weight_difference, weight_sum, double_m2;

    // Temp vector to store collision velocities sum, no need to access global memory at each iteration
    float3 v_temp = {0.0f, 0.0f, 0.0f};

    bool not_colliding;

    // Aux values to save block offsets and real load index
    unsigned block_offset;
    unsigned load_index;

    // Iterate over grid with current block of threads
    for (int block = 0; block < gridDim.x; block++) {
        // Update indexes
        block_offset = block * blockDim.x;
        load_index = block_offset + threadIdx.x;

        // Load data to shared memory
        shared_pos[threadIdx.x] = p_curr.pos[load_index];
        shared_vel[threadIdx.x] = p_curr.vel[load_index];

        // Ensure all values are in shared memory
        __syncthreads();

        // Iterate over all others elements in the block
        for (int i = 0; i < blockDim.x; i++) {
            // Load particle_2 data
            pos_p2 = shared_pos[i];
            vel_p2 = shared_vel[i];

            // Calculate per axis distance
            // In case of collision, distance are only used to calculate Euclidean distance,
            // thus they can be kept in reversed order
            dx = pos_p2.x - pos_p1.x;
            dy = pos_p2.y - pos_p1.y;
            dz = pos_p2.z - pos_p1.z;

            // Calculate inverse Euclidean distance between two particles, get rid of division
            ir = rsqrt(dx * dx + dy * dy + dz * dz);

            // If out of bounds simply set p2_weight to 0 -> which ensures both velocities fill be 0
            pos_p2.w = block_offset + i < N && global_id < N ? pos_p2.w : 0.0f;

            // Save values below to registers to save accesses to memory and multiple calculations of same code
            ir3 = ir * ir * ir + FLT_MIN;
            not_colliding = ir < COLLISION_DISTANCE_INVERSE;
            weight_difference = pos_p1.w - pos_p2.w;
            weight_sum = pos_p1.w + pos_p2.w;
            double_m2 = pos_p2.w * 2.0f;
            Fg_dt_m2_r = G * dt * ir3 * pos_p2.w;

            // If there is collision add collision velocities, otherwise gravitational ->
            // gravitational velocities are skipped if there is collision, likewise vice versa
            v_temp.x += not_colliding ? Fg_dt_m2_r * dx :
                        ((vel_p1.x * weight_difference + double_m2 * vel_p2.x) / weight_sum) - vel_p1.x;
            v_temp.y += not_colliding ? Fg_dt_m2_r * dy :
                        ((vel_p1.y * weight_difference + double_m2 * vel_p2.y) / weight_sum) - vel_p1.y;
            v_temp.z += not_colliding ? Fg_dt_m2_r * dz :
                        ((vel_p1.z * weight_difference + double_m2 * vel_p2.z) / weight_sum) - vel_p1.z;
        }
        // Ensures no thread loads new value to shared until all warps are ready
        __syncthreads();
    }

    // Skip if thread is out of range
    if (global_id < N) {
        // Update values in global context
        vel_p1.x += v_temp.x;
        vel_p1.y += v_temp.y;
        vel_p1.z += v_temp.z;
        p_next.vel[global_id] = vel_p1;

        // Positions are updated with current velocities
        pos_p1.x += vel_p1.x * dt;
        pos_p1.y += vel_p1.y * dt;
        pos_p1.z += vel_p1.z * dt;
        p_next.pos[global_id] = pos_p1;
    }

}// end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------


__global__ void
centerOfMass(t_particles p, float * volatile comX, float *comY, float *comZ, float *comW, int *lock, const int N) {
    extern __shared__ float shared[];
    // Reinterpret shared memory to float4*, for simple indexing
    auto *shared_pos = reinterpret_cast<float4 *>(shared);


    unsigned global_id = threadIdx.x + blockIdx.x * blockDim.x;
    // share memory index
    unsigned shared_index = threadIdx.x;

    // Temp vector to store reduction of sum (x*w, y*w,z*w,w) per block, no need to access global memory at each iteration
    float4 temp_pos = {0.0f, 0.0f, 0.0f, 0.0f};

    // Temp vector to calculate normalized positions, divide by sum of w
    float4 temp_normalized;

    // Temp vector to store vector differences
    float4 diff;

    // Temp vector to store particle to reduce with
    float4 reduce_with;

    // Iterate over excess of data which does not have corresponding thread and reduce them to temporal vector
    for (; global_id < N; global_id += blockDim.x * gridDim.x) {
        reduce_with = p.pos[global_id];
        // Multiply position by weight to avoid unnecessary scaling
        temp_pos.w += reduce_with.w;
        temp_pos.x += reduce_with.x * temp_pos.w;
        temp_pos.y += reduce_with.y * temp_pos.w;
        temp_pos.z += reduce_with.z * temp_pos.w;
    }

    // Store reduced values to shared memory
    shared_pos[shared_index] = temp_pos;

    // Wait for all threads to finish
    __syncthreads();

    // Apply reduction over block, disabling warps not blocks, smaller divergence
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (shared_index < stride) {
            shared_pos[shared_index].x += shared_pos[shared_index + stride].x;
            shared_pos[shared_index].y += shared_pos[shared_index + stride].y;
            shared_pos[shared_index].z += shared_pos[shared_index + stride].z;
            shared_pos[shared_index].w += shared_pos[shared_index + stride].w;
        }
        // Wait for all threads before next reduction
        __syncthreads();
    }
    // Wait for all threads, reduced value is saved in the 0-th index of shared memory
    __syncthreads();

    // Single thread per block writes to output
    if (threadIdx.x == 0) {

        // Normalize positions by diving by weight (this approach simplifies whole computation)
        temp_normalized = shared_pos[0];
        temp_normalized.x /= temp_normalized.w;
        temp_normalized.y /= temp_normalized.w;
        temp_normalized.z /= temp_normalized.w;

        // Spinlock - wait for critical section
        while (atomicExch(lock, 1) != 0);

        // Calculate position difference with current center of mass
        diff.x = temp_normalized.x - *comX;
        diff.y = temp_normalized.y - *comY;
        diff.z = temp_normalized.z - *comZ;

        // Calculate weight ratio only if at least one particle isn't massless
        diff.w = ((temp_normalized.w + *comW) > 0.0f)
                 ? (temp_normalized.w / (temp_normalized.w + *comW)) : 0.0f;

        // Update position and weight of the center-of-mass according to the weight ration and vector
        *comX += diff.x * diff.w;
        *comY += diff.y * diff.w;
        *comZ += diff.z * diff.w;
        *comW += temp_normalized.w;

        // Clean up cache if n_blocks > SM, and concurrency is enabled
        __threadfence();

        // Leave critical section
        atomicExch(lock, 0);
    }
}// end of centerOfMass
//----------------------------------------------------------------------------------------------------------------------

/**
 * CPU implementation of the Center of Mass calculation
 * @param particles - All particles in the system
 * @param N         - Number of particles
 */
__host__ float4 centerOfMassCPU(MemDesc &memDesc) {
    float4 com = {0, 0, 0, 0};

    for (int i = 0; i < memDesc.getDataSize(); i++) {
        // Calculate the vector on the line connecting current body and most recent position of center-of-mass
        const float dx = memDesc.getPosX(i) - com.x;
        const float dy = memDesc.getPosY(i) - com.y;
        const float dz = memDesc.getPosZ(i) - com.z;

        // Calculate weight ratio only if at least one particle isn't massless
        const float dw = ((memDesc.getWeight(i) + com.w) > 0.0f)
                         ? (memDesc.getWeight(i) / (memDesc.getWeight(i) + com.w)) : 0.0f;

        // Update position and weight of the center-of-mass according to the weight ration and vector
        com.x += dx * dw;
        com.y += dy * dw;
        com.z += dz * dw;
        com.w += memDesc.getWeight(i);
    }
    return com;
}// enf of centerOfMassCPU
//----------------------------------------------------------------------------------------------------------------------
