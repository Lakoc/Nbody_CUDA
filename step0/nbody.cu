/**
 * @File nbody.cu
 *
 * Implementation of the N-Body problem
 *
 * Paralelní programování na GPU (PCG 2022)
 * Projekt c. 1 (cuda)
 * Login: xlogin00
 */

#include <cmath>
#include <cfloat>
#include "nbody.h"

/**
 * CUDA kernel to calculate gravitation velocity
 * @param p       - particles
 * @param tmp_vel - temp array for velocities
 * @param N       - Number of particles
 * @param dt      - Size of the time step
 */
__global__ void calculate_gravitation_velocity(t_particles p, t_velocities tmp_vel, int N, float dt) {
    unsigned global_id = threadIdx.x + blockIdx.x * blockDim.x;
    float pos_x = p.pos_x[global_id];
    float pos_y = p.pos_y[global_id];
    float pos_z = p.pos_z[global_id];

    float r, dx, dy, dz, r3, G_dt_r3, Fg_dt_m2_r;;
    float vx = 0.0f;
    float vy = 0.0f;
    float vz = 0.0f;
    bool not_colliding;

    for (int i = 0; i < N; i++) {
        dx = p.pos_x[i] - pos_x;
        dy = p.pos_y[i] - pos_y;
        dz = p.pos_z[i] - pos_z;

        r = sqrt(dx * dx + dy * dy + dz * dz);
        not_colliding = r > COLLISION_DISTANCE;

        r3 = r * r * r + FLT_MIN;

        // Fg*dt/m1/r = G*m1*m2*dt / r^3 / m1 = G*dt/r^3 * m2
        G_dt_r3 = G * dt / r3;
        Fg_dt_m2_r = G_dt_r3 * p.weight[i];


        // vx = - Fx*dt/m2 = - Fg*dt/m2 * dx/r = - Fg*dt/m2/r * dx
        vx += not_colliding ? Fg_dt_m2_r * dx : 0.0f;
        vy += not_colliding ? Fg_dt_m2_r * dy : 0.0f;
        vz += not_colliding ? Fg_dt_m2_r * dz : 0.0f;
    }
    tmp_vel.x[global_id] = vx;
    tmp_vel.y[global_id] = vy;
    tmp_vel.z[global_id] = vz;

}// end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to calculate collision velocity
 * @param p       - particles
 * @param tmp_vel - temp array for velocities
 * @param N       - Number of particles
 * @param dt      - Size of the time step
 */
__global__ void calculate_collision_velocity(t_particles p, t_velocities tmp_vel, int N, float dt) {
    unsigned global_id = threadIdx.x + blockIdx.x * blockDim.x;
    float pos_x = p.pos_x[global_id];
    float pos_y = p.pos_y[global_id];
    float pos_z = p.pos_z[global_id];
    float weight = p.weight[global_id];
    float vel_x = p.vel_x[global_id];
    float vel_y = p.vel_y[global_id];
    float vel_z = p.vel_z[global_id];


    float r, dx, dy, dz, p2_weight, m1_minus_m2, m1_plus_m2, double_m2;
    float vx = 0.0f;
    float vy = 0.0f;
    float vz = 0.0f;
    bool colliding;

    for (int i = 0; i < N; i++) {
        dx = pos_x - p.pos_x[i];
        dy = pos_y - p.pos_y[i];
        dz = pos_z - p.pos_z[i];

        r = sqrt(dx * dx + dy * dy + dz * dz);
        p2_weight = p.weight[i];
        colliding = r > 0.0f && r < COLLISION_DISTANCE;
        m1_minus_m2 = weight - p2_weight;
        m1_plus_m2 = weight + p2_weight;
        double_m2 = p2_weight * 2.0;
        vx += colliding ? ((vel_x * m1_minus_m2 + double_m2 * p.vel_x[i]) / m1_plus_m2) - vel_x : 0.0f;
        vy += colliding ? ((vel_y * m1_minus_m2 + double_m2 * p.vel_y[i]) / m1_plus_m2) - vel_y : 0.0f;
        vz += colliding ? ((vel_z * m1_minus_m2 + double_m2 * p.vel_z[i]) / m1_plus_m2) - vel_z : 0.0f;
    }

    tmp_vel.x[global_id] += vx;
    tmp_vel.y[global_id] += vy;
    tmp_vel.z[global_id] += vz;
}// end of calculate_collision_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to update particles
 * @param p       - particles
 * @param tmp_vel - temp array for velocities
 * @param N       - Number of particles
 * @param dt      - Size of the time step
 */
__global__ void update_particle(t_particles p, t_velocities tmp_vel, int N, float dt) {
    unsigned global_id = threadIdx.x + blockIdx.x * blockDim.x;
    p.vel_x[global_id] += tmp_vel.x[global_id];
    p.vel_y[global_id] += tmp_vel.y[global_id];
    p.vel_z[global_id] += tmp_vel.z[global_id];
    p.pos_x[global_id] += p.vel_x[global_id] * dt;
    p.pos_y[global_id] += p.vel_y[global_id] * dt;
    p.pos_z[global_id] += p.vel_z[global_id] * dt;
}// end of update_particle
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to update particles
 * @param p       - particles
 * @param comX    - pointer to a center of mass position in X
 * @param comY    - pointer to a center of mass position in Y
 * @param comZ    - pointer to a center of mass position in Z
 * @param comW    - pointer to a center of mass weight
 * @param lock    - pointer to a user-implemented lock
 * @param N       - Number of particles
 */
__global__ void
centerOfMass(t_particles p, float *comX, float *comY, float *comZ, float *comW, int *lock, const int N) {

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
