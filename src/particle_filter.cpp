/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  //Set the number of particles
  num_particles = 120;

  //Create normal distribution for x, y and theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  //Generate particles with Gaussian distribution
  default_random_engine gen;
  for(int i = 0; i < num_particles; i++){
     Particle temp;
     temp.id = i;
     temp.x = dist_x(gen);
     temp.y = dist_y(gen);
     temp.theta = dist_theta(gen);
     temp.weight = 1;
     particles.push_back(temp);
     weights.push_back(1);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  //Create normal distribution for x, y and theta
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  //Predict particle positions with Gaussian noise
  default_random_engine gen;
  for(int i = 0; i < num_particles; i++){
    double theta = particles[i].theta;
    if (fabs(yaw_rate) < 0.000001) {
      particles[i].x += velocity * delta_t * std::cos(theta);
      particles[i].y += velocity * delta_t * std::sin(theta);
    }
    else {
      particles[i].x += velocity / yaw_rate * (std::sin(theta + yaw_rate * delta_t) - std::sin(theta) );
      particles[i].y += velocity / yaw_rate * (std::cos(theta) - std::cos(theta + yaw_rate * delta_t) );
      particles[i].theta += yaw_rate * delta_t;
      // add Gaussian nose
      particles[i].x += dist_x(gen);
      particles[i].y += dist_y(gen);
      particles[i].theta += dist_theta(gen);
    }
  }

}

void ParticleFilter::getMapROI(double sensor_range, Particle p, const Map &map, std::vector<LandmarkObs>& map_ROI){

  std::vector<Map::single_landmark_s> landmark_list = map.landmark_list;
  for(unsigned int i = 0; i < landmark_list.size(); i++){
    double distance = dist(p.x, p.y, double(landmark_list[i].x_f), double(landmark_list[i].y_f));
    LandmarkObs temp;
    if(distance <= sensor_range){
      temp.id  = landmark_list[i].id_i;
      temp.x   = landmark_list[i].x_f;
      temp.y   = landmark_list[i].y_f;
    }
    map_ROI.push_back(temp);
  }
}

void ParticleFilter::vehicle2map(Particle p, const std::vector<LandmarkObs> &observations, std::vector<LandmarkObs> &map_observations){

  for(unsigned int i = 0; i < observations.size(); i++){
    LandmarkObs temp;
    temp.id = observations[i].id;
    temp.x  = p.x + std::cos(p.theta) * observations[i].x - std::sin(p.theta) * observations[i].y;
    temp.y  = p.y + std::sin(p.theta) * observations[i].x + std::cos(p.theta) * observations[i].y;
    map_observations.push_back(temp);
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> map_ROI, std::vector<LandmarkObs>& observations) {

  for (int i = 0; i < observations.size(); i++) {
    // init minimum distance to maximum possible
    double min_dist = numeric_limits<double>::max();
    // init id of landmark from map
    int map_id = -1;
    // find the map landmark nearest to the current observed landmark
    for (int j = 0; j < map_ROI.size(); j++) {
      double distance = dist(observations[i].x, observations[i].y, map_ROI[j].x, map_ROI[j].y);
      if (distance < min_dist) {
        min_dist = distance;
        map_id = map_ROI[j].id;
      }
    }
    observations[i].id = map_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  for(int i = 0; i < num_particles; i++){
     std::vector<LandmarkObs> map_ROI;
     getMapROI(sensor_range, particles[i], map_landmarks, map_ROI);
     std::vector<LandmarkObs> map_observations;
     vehicle2map(particles[i], observations, map_observations);
     dataAssociation(map_ROI, map_observations);
     //multi-variable Gaussian distribution
     weights[i] = 1.;
     for(int j=0; j < map_observations.size(); j++){
       double gauss_norm = 1./ (2. * M_PI * std_landmark[0] * std_landmark[1]);
       double x_obs = map_observations[j].x;
       double y_obs = map_observations[j].y;
       int map_id = map_observations[j].id;
       double x_map;
       double y_map;
       for (unsigned int k = 0; k < map_ROI.size(); k++) {
         if (map_ROI[k].id == map_id) {
           x_map = map_ROI[k].x;
           y_map = map_ROI[k].y;
           break;
         }
       }
       double exponent = (x_obs-x_map)*(x_obs-x_map) / (2. * std_landmark[0] * std_landmark[0]) + (y_obs-y_map)*(y_obs-y_map) / (2. * std_landmark[1] * std_landmark[1]);
       weights[i] *= gauss_norm * std::exp(-exponent);
     }
     particles[i].weight = weights[i];
  }


}

void ParticleFilter::resample() {
  std::vector<Particle> new_particles;
  default_random_engine gen;
  uniform_int_distribution<int> uni_int_dist(0, num_particles-1);
  int index = uni_int_dist(gen);
  uniform_real_distribution<double> uni_double_dist(0., 1.);
  // Get max weight
  double w_max = *std::max_element(weights.begin(), weights.end());
  double beta = 0.;
  for(unsigned int i = 0; i < num_particles; i++){
    beta += 2.0 * uni_double_dist(gen) * w_max;
    while( weights[index] < beta){
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }
  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
