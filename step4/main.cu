/**
 * @File main.cu
 *
 * The main file of the project
 *
 * Paralelní programování na GPU (PCG 2022)
 * Projekt c. 1 (cuda)
 * Login: xpolok03
 */

#include <sys/time.h>
#include <cstdio>
#include <cmath>

#include "nbody.h"
#include "h5Helper.h"

/**
 * Aux function to log last cuda err
 * Author: Robert Crovella, talonmies  Source: https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 */
#define gpuErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/**
 * Main rotine
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv) {
    // Time measurement
    struct timeval t1, t2;

    if (argc != 10) {
        printf("Usage: nbody <N> <dt> <steps> <threads/block> <write intesity> <reduction threads> <reduction threads/block> <input> <output>\n");
        exit(1);
    }

    // Number of particles
    const int N = std::stoi(argv[1]);
    // Length of time step
    const float dt = std::stof(argv[2]);
    // Number of steps
    const int steps = std::stoi(argv[3]);
    // Number of thread blocks
    const int thr_blc = std::stoi(argv[4]);
    // Write frequency
    int writeFreq = std::stoi(argv[5]);
    // number of reduction threads
    const int red_thr = std::stoi(argv[6]);
    // Number of reduction threads/blocks
    const int red_thr_blc = std::stoi(argv[7]);

    // Size of the simulation CUDA gird - number of blocks
    const size_t simulationGrid = (N + thr_blc - 1) / thr_blc;
    // Size of the reduction CUDA grid - number of blocks
    const size_t reductionGrid = (red_thr + red_thr_blc - 1) / red_thr_blc;

    // Log benchmark setup
    printf("N: %d\n", N);
    printf("dt: %f\n", dt);
    printf("steps: %d\n", steps);
    printf("threads/block: %d\n", thr_blc);
    printf("blocks/grid: %lu\n", simulationGrid);
    printf("reduction threads/block: %d\n", red_thr_blc);
    printf("reduction blocks/grid: %lu\n", reductionGrid);

    const size_t recordsNum = (writeFreq > 0) ? (steps + writeFreq - 1) / writeFreq : 0;
    writeFreq = (writeFreq > 0) ? writeFreq : 0;


    t_particles particles_cpu;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                            FILL IN: CPU side memory allocation (step 0)                                          //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    size_t particles_pos_arr_size = N * sizeof(float4);
    size_t particles_vel_arr_size = N * sizeof(float3);

    // Allocation of pinned, pageable memory at host
    gpuErrCheck(cudaMallocHost(&particles_cpu.pos, particles_pos_arr_size));
    gpuErrCheck(cudaMallocHost(&particles_cpu.vel, particles_vel_arr_size));


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                              FILL IN: memory layout descriptor (step 0)                                          //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /*
     * Caution! Create only after CPU side allocation
     * parameters:
     *                      Stride of two               Offset of the first
     *  Data pointer        consecutive elements        element in floats,
     *                      in floats, not bytes        not bytes
    */
    MemDesc md(
            &particles_cpu.pos->x, 4, 0,              // Postition in X
            &particles_cpu.pos->y, 4, 0,              // Postition in Y
            &particles_cpu.pos->z, 4, 0,              // Postition in Z
            &particles_cpu.vel->x, 3, 0,              // Velocity in X
            &particles_cpu.vel->y, 3, 0,              // Velocity in Y
            &particles_cpu.vel->z, 3, 0,              // Velocity in Z
            &particles_cpu.pos->w, 4, 0,              // Weight
            N,                                                                  // Number of particles
            recordsNum);                                                              // Number of records in output file

    // Initialisation of helper class and loading of input data
    H5Helper h5Helper(argv[8], argv[9], md);

    try {
        h5Helper.init();
        h5Helper.readParticleData();
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }


    t_particles particles_gpu_curr;
    t_particles particles_gpu_next;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                  FILL IN: GPU side memory allocation (step 0)                                    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Allocation of current and next particles data
    gpuErrCheck(cudaMalloc<float4>(&particles_gpu_curr.pos, particles_pos_arr_size))
    gpuErrCheck(cudaMalloc<float3>(&particles_gpu_curr.vel, particles_vel_arr_size))

    gpuErrCheck(cudaMalloc<float4>(&particles_gpu_next.pos, particles_pos_arr_size))
    gpuErrCheck(cudaMalloc<float3>(&particles_gpu_next.vel, particles_vel_arr_size))

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                       FILL IN: memory transfers (step 0)                                         //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Copy input data to gpu memory
    gpuErrCheck(cudaMemcpy(particles_gpu_curr.pos, particles_cpu.pos, particles_pos_arr_size, cudaMemcpyHostToDevice))
    gpuErrCheck(cudaMemcpy(particles_gpu_curr.vel, particles_cpu.vel, particles_vel_arr_size, cudaMemcpyHostToDevice))

    gpuErrCheck(cudaMemcpy(particles_gpu_next.pos, particles_cpu.pos, particles_vel_arr_size, cudaMemcpyHostToDevice))
    // There is no need to copy velocities to next state of particles, they will never be accessed until calculation of next step


    dim3 dimBlock(thr_blc);
    dim3 dimGrid(simulationGrid);

    // Size of shared memory to keep whole data in current shared mem over block
    size_t sharedMemory = thr_blc * (sizeof(float4) + sizeof(float3));


    float4 *comGPU;
    int *lock;

    // Allocate memory for center of mass calculation
    gpuErrCheck(cudaMalloc(&comGPU, sizeof(float4)))
    gpuErrCheck(cudaMemset(comGPU, 0, sizeof(float4)))
    gpuErrCheck(cudaMalloc(&lock, sizeof(int)))
    gpuErrCheck(cudaMemset(lock, 0, sizeof(int)))

    // Calculate share memory size for reduction of center of mass
    size_t cmo_shared_size = thr_blc * sizeof(float) * 4;

    // Allocate pinned memory for center of mass computed in loop
    float4 *comCPU;
    gpuErrCheck(cudaMallocHost(&comCPU, sizeof(float4)))

    gettimeofday(&t1, 0);

    size_t record_num = 0;

    // Declare and initialize center of mass and velocity calculation streams and events
    cudaStream_t velocity_stream, com_stream;
    cudaEvent_t com_updated, com_copied, particles_updated, particles_copied;

    gpuErrCheck(cudaStreamCreate(&velocity_stream))
    gpuErrCheck(cudaStreamCreate(&com_stream))

    gpuErrCheck(cudaEventCreate(&com_updated))
    gpuErrCheck(cudaEventCreate(&com_copied))
    gpuErrCheck(cudaEventCreate(&particles_updated))
    gpuErrCheck(cudaEventCreate(&particles_copied))


    // Replacement of std::swap with indexing mod step -> synchronization over epochs would be necessary with std::swap
    t_particles *particles[2] = {&particles_gpu_curr, &particles_gpu_next};

    for (int s = 0; s < steps; s++) {
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //                                       FILL IN: kernels invocation (step 0)                                     //
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Calculation of t[n+1] particle_data has to wait for t[n] center of mass calculation (3.5.1. IV)
        gpuErrCheck(cudaStreamWaitEvent(velocity_stream, com_updated))
        calculate_velocity<<<dimGrid, dimBlock, sharedMemory, velocity_stream>>>(*particles[s % 2],
                                                                                 *particles[(s + 1) % 2], N, dt);

        // Event to enable writing particle_data to output file (3.5.1. VI)
        gpuErrCheck(cudaEventRecord(particles_updated, velocity_stream))

        // Calculation of t[n] center of mass has to wait for t[n-1] particle_data calculation (3.5.1. V)
        gpuErrCheck(cudaStreamWaitEvent(com_stream, particles_updated))

        // Clear center of mass data from last epoch
        gpuErrCheck(cudaMemsetAsync(comGPU, 0, sizeof(float4), com_stream))
        centerOfMass<<<dimGrid, dimBlock, cmo_shared_size, com_stream>>>(*particles[(s) % 2], &comGPU->x, &comGPU->y,
                                                                         &comGPU->z,
                                                                         &comGPU->w, lock, N);

        // Event to enable start of particle_data calculation for t[n+1] (3.5.1. IV)
        gpuErrCheck(cudaEventRecord(com_updated, com_stream))


        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //                                          FILL IN: synchronization  (step 4)                                    //
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        if (writeFreq > 0 && (s % writeFreq == 0)) {
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //                          FILL IN: synchronization and file access logic (step 4)                             //
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            // Default synchronization after t[n] particle_data calculation is preserved (3.5.1. III)
            gpuErrCheck(cudaMemcpyAsync(particles_cpu.pos, (*particles[s % 2]).pos, particles_pos_arr_size,
                                        cudaMemcpyDeviceToHost, velocity_stream))
            gpuErrCheck(cudaMemcpyAsync(particles_cpu.vel, (*particles[s % 2]).vel, particles_vel_arr_size,
                                        cudaMemcpyDeviceToHost, velocity_stream))

            // Event to enable writing particle_data to output file (3.5.1. III)
            gpuErrCheck(cudaEventRecord(particles_copied, velocity_stream))

            // Writing to disk of particle_data t[n] has to wait for t[n-1] particle_data calculation (3.5.1. VI)
            gpuErrCheck(cudaStreamWaitEvent(velocity_stream, particles_updated))

            // Writing to disk has to wait for copy to finish (3.5.1. VIII)
            // There is no synchronization in following lines, thus copying in parallel with writing to stdout is enabled (3.5.1. VIII)
            gpuErrCheck(cudaEventSynchronize(particles_copied))
            h5Helper.writeParticleData(record_num);

            // Default synchronization after t[n] center of mass calculation is preserved (3.5.1. VII)
            gpuErrCheck(cudaMemcpyAsync(comCPU, comGPU, sizeof(float4), cudaMemcpyDeviceToHost, com_stream))

            // Event to enable writing particle_data to output file (3.5.1. III)
            gpuErrCheck(cudaEventRecord(com_copied, com_stream))

            // Writing to disk has to wait for copy to finish (3.5.1. IX -> derived not directly)
            gpuErrCheck(cudaEventSynchronize(com_copied))
            h5Helper.writeCom(comCPU->x, comCPU->y, comCPU->z, comCPU->w, record_num);

            // Increment record number
            record_num += 1;
        }
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //              FILL IN: invocation of center-of-mass kernel (step 3.1, step 3.2, step 4)                           //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    gpuErrCheck(cudaDeviceSynchronize())

    gpuErrCheck(cudaPeekAtLastError())

    gpuErrCheck(cudaMemset(comGPU, 0, sizeof(float4)))

    centerOfMass<<<dimGrid, dimBlock, cmo_shared_size>>>(particles_gpu_curr, &comGPU->x, &comGPU->y, &comGPU->z,
                                                         &comGPU->w, lock, N);


    gettimeofday(&t2, 0);

    // Approximate simulation wall time
    double t = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000000.0;
    printf("Time: %f s\n", t);


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                             FILL IN: memory transfers for particle data (step 0)                                 //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    float4 comOnGPU;

    gpuErrCheck(cudaMemcpy(particles_cpu.pos, particles_gpu_curr.pos, particles_pos_arr_size, cudaMemcpyDeviceToHost))
    gpuErrCheck(cudaMemcpy(particles_cpu.vel, particles_gpu_curr.vel, particles_vel_arr_size, cudaMemcpyDeviceToHost))


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                        FILL IN: memory transfers for center-of-mass (step 3.1, step 3.2)                         //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    gpuErrCheck(cudaMemcpy(&comOnGPU, comGPU, sizeof(float4), cudaMemcpyDeviceToHost))

    float4 comOnCPU = centerOfMassCPU(md);

    std::cout << "Center of mass on CPU:" << std::endl
              << comOnCPU.x << ", "
              << comOnCPU.y << ", "
              << comOnCPU.z << ", "
              << comOnCPU.w
              << std::endl;

    std::cout << "Center of mass on GPU:" << std::endl
              << comOnGPU.x << ", "
              << comOnGPU.y << ", "
              << comOnGPU.z << ", "
              << comOnGPU.w
              << std::endl;

    // Writing final values to the file
    h5Helper.writeComFinal(comOnGPU.x, comOnGPU.y, comOnGPU.z, comOnGPU.w);
    h5Helper.writeParticleDataFinal();

    gpuErrCheck(cudaFreeHost(particles_cpu.pos))
    gpuErrCheck(cudaFreeHost(particles_cpu.vel))
    gpuErrCheck(cudaFreeHost(comCPU))

    gpuErrCheck(cudaFree(particles_gpu_curr.pos))
    gpuErrCheck(cudaFree(particles_gpu_curr.vel))
    gpuErrCheck(cudaFree(particles_gpu_next.pos))
    gpuErrCheck(cudaFree(particles_gpu_next.vel))

    gpuErrCheck(cudaFree(comGPU))
    gpuErrCheck(cudaFree(lock))

    gpuErrCheck(cudaStreamDestroy(velocity_stream))
    gpuErrCheck(cudaStreamDestroy(com_stream))

    gpuErrCheck(cudaEventDestroy(com_updated))
    gpuErrCheck(cudaEventDestroy(com_copied))
    gpuErrCheck(cudaEventDestroy(particles_updated))
    gpuErrCheck(cudaEventDestroy(particles_copied))

    return 0;
}// end of main
//----------------------------------------------------------------------------------------------------------------------
