#include <stdio.h>
#include <math.h>

#define R 6378.1
#define TO_RAD (M_PI / 180)

__global__ void calc_distance(double *latitude, double *longitude, double *t_latitude, double *t_longitude, double *result_distance){
    double lat, lng, t_lat, t_lng;
    lat = *latitude;
    lng = *longitude;
    t_lat = *t_latitude;
    t_lng = *t_longitude;

    lng -= t_lng;
    lng *= TO_RAD;
    lat *= TO_RAD;
    t_lat *= TO_RAD;
    double dx, dy, dz;
    dz = sin(lat) - sin(t_lat);
    dx = cos(lng) * cos(lat) - cos(t_lat);
    dy = sin(lng) * cos(lat);
    double distance =  asin(sqrt(dx * dx + dy * dy + dz * dz) / 2) * 2 * R;
    *result_distance = distance;
}

int main(void){
    //double geo[2][2] = {{35, 150}, {35, 145}};
    //double tweet[2][2] = {{40, 135}, {30, 140}};
    double geo_lat[2] = {35, 35};
    double geo_lng[2] = {150, 145};
    double tweet_lat[2] = {40, 30};
    double tweet_lng[2] = {135, 140};
    int geo_size = sizeof(geo_lat);
    int tweet_size = sizeof(tweet_lat);
    int distance_size = geo_size * tweet_size;

    double *distance;
    distance = (double *)malloc(distance_size * sizeof(double));
    
    double *d_geo_lat, *d_geo_lng, *d_tweet_lat, *d_tweet_lng, *d_distance;
    cudaMalloc(&d_geo_lat, geo_size * sizeof(double));
    cudaMalloc(&d_geo_lng, geo_size * sizeof(double));
    cudaMalloc(&d_tweet_lat, tweet_size * sizeof(double));
    cudaMalloc(&d_tweet_lng, tweet_size * sizeof(double));
    cudaMalloc(&d_distance, distance_size * sizeof(double));

    // copy data from host memory to GPU memory
    cudaMemcpy(d_geo_lat, geo_lat, geo_size * sizeof(double), cudaMemcpyHostToDevice ); 
    cudaMemcpy(d_geo_lng, geo_lng, geo_size * sizeof(double), cudaMemcpyHostToDevice ); 
    cudaMemcpy(d_tweet_lat, tweet_lat, tweet_size * sizeof(double), cudaMemcpyHostToDevice ); 
    cudaMemcpy(d_tweet_lng, tweet_lng, tweet_size * sizeof(double), cudaMemcpyHostToDevice ); 
    cudaMemcpy(d_distance, distance, distance_size * sizeof(double), cudaMemcpyHostToDevice ); 

    calc_distance(d_geo_lat, d_geo_lng, d_tweet_lat, d_tweet_lng, d_distance);

    cudaMemcpy(distance, d_distance, geo_size * sizeof(double), cudaMemcpyDeviceToHost );

    free(distance);
    cudaFree(d_geo_lat);
    cudaFree(d_geo_lng);
    cudaFree(d_tweet_lat);
    cudaFree(d_tweet_lng);
    cudaFree(d_distance);
    return 0;
}
