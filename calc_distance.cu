#include <stdio.h>
#include <math.h>

#define TO_RAD (M_PI / 180)

__global__ void calc_distance(double *latitude, double *longitude, double *time, double *t_latitude, double *t_longitude, double *t_time, double *t_flag, double *result_distance, double *time_lag, double *tweet_flag, int nx){
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    double lat, lng, t_lat, t_lng;
    lat = latitude[ix];
    lng = longitude[ix];
    t_lat = t_latitude[iy];
    t_lng = t_longitude[iy];

    lng -= t_lng;
    lng *= TO_RAD;
    lat *= TO_RAD;
    t_lat *= TO_RAD;
    double dx, dy, dz;
    dz = sin(lat) - sin(t_lat);
    dx = cos(lng) * cos(lat) - cos(t_lat);
    dy = sin(lng) * cos(lat);
    // Earth radius is 6371 km
    double distance =  asin(sqrt(dx * dx + dy * dy + dz * dz) / 2) * 2 * 6371;
    result_distance[ix + iy * nx] = distance;

    time_lag[ix + iy * nx] = fabs(time[ix] - t_time[iy]);
    tweet_flag[ix + iy * nx] = t_flag[iy];
    printf("threadIdx.x %d, threadIdx.y %d, ix %d, iy %d, distance idx %d, distance %f, time_lag %f, time_flag %f\n", threadIdx.x, threadIdx.y, ix, iy, ix + iy * nx, distance, time_lag[ix + iy * nx], t_flag[iy]);
}


// calculate distance between geo and tweet by GPU 
double* calc_distance2d_gpu(double *geo_lat, double *geo_lng, double *geo_time, double *tweet_lat, double *tweet_lng, double *tweet_time, double *tweet_flag, int geo_size, int tweet_size){
    int distance_size = geo_size * tweet_size;
    double *distance, *time_lag, *return_tweet_flag;
    distance = (double *)malloc(distance_size * sizeof(double));
    time_lag = (double *)malloc(distance_size * sizeof(double));
    return_tweet_flag = (double *)malloc(distance_size * sizeof(double));
    
    double *d_geo_lat, *d_geo_lng, *d_geo_time, *d_tweet_lat, *d_tweet_lng, *d_tweet_time, *d_tweet_flag, *d_distance, *d_time_lag, *d_return_tweet_flag;
    cudaMalloc(&d_geo_lat, geo_size * sizeof(double));
    cudaMalloc(&d_geo_lng, geo_size * sizeof(double));
    cudaMalloc(&d_geo_time, geo_size * sizeof(double));
    cudaMalloc(&d_tweet_lat, tweet_size * sizeof(double));
    cudaMalloc(&d_tweet_lng, tweet_size * sizeof(double));
    cudaMalloc(&d_tweet_time, tweet_size * sizeof(double));
    cudaMalloc(&d_tweet_flag, tweet_size * sizeof(double));
    cudaMalloc(&d_distance, distance_size * sizeof(double));
    cudaMalloc(&d_time_lag, distance_size * sizeof(double));
    cudaMalloc(&d_return_tweet_flag, distance_size * sizeof(double));

    // transfer copy data from host memory to GPU memory
    cudaMemcpy(d_geo_lat, geo_lat, geo_size * sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy(d_geo_lng, geo_lng, geo_size * sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy(d_geo_time, geo_time, geo_size * sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy(d_tweet_lat, tweet_lat, tweet_size * sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy(d_tweet_lng, tweet_lng, tweet_size * sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy(d_tweet_time, tweet_time, tweet_size * sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy(d_tweet_flag, tweet_flag, tweet_size * sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy(d_distance, distance, distance_size * sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy(d_time_lag, time_lag, distance_size * sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy(d_return_tweet_flag, return_tweet_flag, distance_size * sizeof(double), cudaMemcpyHostToDevice );


    //int nx = 2;
    //int ny = 2;
    dim3 block(geo_size, tweet_size);
    dim3 grid(1);
    int nx = geo_size;
    calc_distance<<<grid, block>>>(d_geo_lat, d_geo_lng, d_geo_time, d_tweet_lat, d_tweet_lng, d_tweet_time, d_tweet_flag, d_distance, d_time_lag, d_return_tweet_flag, nx);

    cudaMemcpy(distance, d_distance, distance_size * sizeof(double), cudaMemcpyDeviceToHost );
    cudaMemcpy(time_lag, d_time_lag, distance_size * sizeof(double), cudaMemcpyDeviceToHost );
    cudaMemcpy(return_tweet_flag, d_return_tweet_flag, distance_size * sizeof(double), cudaMemcpyDeviceToHost );

    // free memory
    cudaFree(d_geo_lat);
    cudaFree(d_geo_lng);
    cudaFree(d_geo_time);
    cudaFree(d_tweet_lat);
    cudaFree(d_tweet_lng);
    cudaFree(d_tweet_time);
    cudaFree(d_tweet_flag);
    cudaFree(d_distance);
    cudaFree(d_time_lag);
    cudaFree(d_tweet_flag);
    return distance;
}


int main(void){
    //double geo[2][2] = {{35, 150}, {35, 145}};
    //double tweet[2][2] = {{40, 135}, {30, 140}};
    int geo_size = 5;
    int tweet_size = 2;
    // geo parameter
    double geo_lat[5] = {35, 30, 45, 32, 33};
    double geo_lng[5] = {150, 145, 130, 140, 141};
    double geo_time[5] = {1, 2, 3, 4, 5};
    // time parameter
    double tweet_lat[2] = {40, 50};
    double tweet_lng[2] = {135, 140};
    double tweet_time[2] = {1, 2};
    double tweet_flag[2] = {0, 1};

    double *distance_matrix;
    distance_matrix = calc_distance2d_gpu(geo_lat, geo_lng, geo_time, tweet_lat, tweet_lng, tweet_time, tweet_flag, geo_size, tweet_size);
    printf("returned\n");
    for(int i=0; i < geo_size * tweet_size; i++){
        printf("distance %f\n", distance_matrix[i]);
    }
    free(distance_matrix);
    return 0;
}
