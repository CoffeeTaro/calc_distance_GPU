#ifndef _CALCULATEDISTANCE_H_
#define _CALCULATEDISTANCE_H_

__global__ void calc_distance(double *latitude, double *longitude, double *time, double *t_latitude, double *t_longitude, double *t_time, double *t_flag, double *result_distance, double *time_lag, double *tweet_flag, int *nx);
void calc_distance2d_gpu(double *geo_lat, double *geo_lng, double *geo_time, double *tweet_lat, double *tweet_lng, double *tweet_time, double *tweet_flag, int geo_size, int tweet_size, double **result);

#endif // _CALCULATEDISTANCE_H_
