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

    // haver shine
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
    //printf("geoIndex %d, twiIndex %d, distance idx %d, distance %f, time_lag %f, time_flag %f\n", ix, iy, ix + iy * nx, distance, time_lag[ix + iy * nx], t_flag[iy]);
}


// calculate distance between geo and tweet by GPU 
void calc_distance2d_gpu(double *geo_lat, double *geo_lng, double *geo_time, double *tweet_lat, double *tweet_lng, double *tweet_time, double *tweet_flag, int geo_size, int tweet_size, double **result){
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

    // configure parallel
    //dim3 block(geo_size, tweet_size);
    //dim3 grid(1);
    dim3 block(1);
    dim3 grid(geo_size, tweet_size);
    int nx = geo_size;
    calc_distance<<<grid, block>>>(d_geo_lat, d_geo_lng, d_geo_time, d_tweet_lat, d_tweet_lng, d_tweet_time, d_tweet_flag, d_distance, d_time_lag, d_return_tweet_flag, nx);

    cudaMemcpy(distance, d_distance, distance_size * sizeof(double), cudaMemcpyDeviceToHost );
    cudaMemcpy(time_lag, d_time_lag, distance_size * sizeof(double), cudaMemcpyDeviceToHost );
    cudaMemcpy(return_tweet_flag, d_return_tweet_flag, distance_size * sizeof(double), cudaMemcpyDeviceToHost );

    /*
    the indexes of returned arrays(pointers) is below.
    geoIndex 2, twiIndex 1, distance idx 7, distance 933.287262, time_lag 1.000000, time_flag 1.000000
    geoIndex 0, twiIndex 1, distance idx 5, distance 1854.648262, time_lag 1.000000, time_flag 1.000000
    geoIndex 1, twiIndex 0, distance idx 1, distance 1435.332879, time_lag 1.000000, time_flag 0.000000
    geoIndex 4, twiIndex 1, distance idx 9, distance 1892.102074, time_lag 3.000000, time_flag 1.000000
    geoIndex 4, twiIndex 0, distance idx 4, distance 944.689913, time_lag 4.000000, time_flag 0.000000
    geoIndex 0, twiIndex 0, distance idx 0, distance 1432.937683, time_lag 0.000000, time_flag 0.000000
    geoIndex 3, twiIndex 1, distance idx 8, distance 2001.508680, time_lag 2.000000, time_flag 1.000000
    geoIndex 1, twiIndex 1, distance idx 6, distance 2263.027381, time_lag 0.000000, time_flag 1.000000
    geoIndex 3, twiIndex 0, distance idx 3, distance 996.360905, time_lag 3.000000, time_flag 0.000000
    geoIndex 2, twiIndex 0, distance idx 2, distance 690.440334, time_lag 2.000000, time_flag 0.000000

    */

    // free memory on GPU
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

    //double dd[5][2][3];
    // 3D array [geo_size][tweet_size][3]
    double ***dd_matrix = (double ***)malloc(sizeof(double **) * geo_size);
    for(int dd1=0; dd1<geo_size; dd1++){
        dd_matrix[dd1] = (double **)malloc(sizeof(double *) * tweet_size); 
        for(int dd2=0; dd2<tweet_size; dd2++){
            dd_matrix[dd1][dd2] = (double *)malloc(sizeof(double) * 3);
        }
    }

    for(int i = 0; i < distance_size; i++){
        int j = i % geo_size;
        int k = i / geo_size;
        dd_matrix[j][k][0] = distance[i];
        dd_matrix[j][k][1] = time_lag[i];
        dd_matrix[j][k][2] = return_tweet_flag[i];
        //printf("j %d, k %d, distance %f\n", j, k, dd_matrix[j][k][0]);
        //printf("j %d, k %d, time_lag %f\n", j, k, dd_matrix[j][k][1]);
        //printf("j %d, k %d, flag %f\n", j, k, dd_matrix[j][k][2]);
    }


    // find shortest distance and save it's distance, it's time_lag, and it's tweet_flag
    for(int g = 0; g < geo_size ; g++){
        double minimum;
        double m_time;
        double m_flag;
        for(int t = 0; t < tweet_size ; t++){
            if(t != 0){
                if(minimum > dd_matrix[g][t][0]){
                    // update
                    minimum = dd_matrix[g][t][0];
                    m_time = dd_matrix[g][t][1];
                    m_flag = dd_matrix[g][t][2];
                }
            }else{
                // initialize
                minimum = dd_matrix[g][t][0]; 
                m_time = dd_matrix[g][t][1];
                m_flag = dd_matrix[g][t][2];
            }
        }
        result[g][0] = minimum;
        result[g][1] = m_time;
        result[g][2] = m_flag;
    }

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

    //int distance_size = geo_size * tweet_size;
    
    double **result_matrix = (double **)malloc(sizeof(double *) * geo_size);
    for(int i=0; i<geo_size; i++){
        result_matrix[i] = (double *)malloc(sizeof(double) * 3); 
    }

    calc_distance2d_gpu(geo_lat, geo_lng, geo_time, tweet_lat, tweet_lng, tweet_time, tweet_flag, geo_size, tweet_size, result_matrix);

    for(int c = 0; c < geo_size; c++){
        printf("minimum %f\n", result_matrix[c][0]);
        printf("time_lag %f\n", result_matrix[c][1]);
        printf("flag %f\n", result_matrix[c][2]);
    }
    free(result_matrix);
    return 0;
}
