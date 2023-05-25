#include <stdio.h>
#include <memory>
#include <vector>
#include <time.h>
#include <algorithm>
#include <pcl/point_types.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/registration/transforms.h>
#define BOOST_TYPEOF_EMULATION
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot.h>
#include <pcl/features/shot_lrf.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/surface/mls.h>
#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>
#include <pcl/visualization/pcl_visualizer.h>
#include<pcl/keypoints/iss_3d.h>
#include "Eva.h"
/*******************************************************************************Descriptor********************************************************/
// void SHOT_compute(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, vector<int> indices, float sup_radius, vector<vector<float>>&features, vector<LRF>&LRFs)
// {
// 	int i, j;
// 	pcl::PointIndicesPtr Idx = boost::shared_ptr<pcl::PointIndices>();
// 	for (j = 0; j<indices.size(); j++)
// 		Idx->indices.push_back(indices[j]);
	
// 	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> n;
// 	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
// 	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
// 	tree->setInputCloud(cloud);
// 	n.setInputCloud(cloud);
// 	n.setSearchMethod(tree);
// 	n.setKSearch(20);
// 	n.compute(*normals);
	
// 	//SHOT_LRF
// 	pcl::PointCloud<pcl::ReferenceFrame>::Ptr pcl_LRF(new pcl::PointCloud<pcl::ReferenceFrame>);
// 	pcl::SHOTLocalReferenceFrameEstimation<pcl::PointXYZ, pcl::ReferenceFrame> LRF_est;
// 	LRF_est.setInputCloud(cloud);
// 	LRF_est.setIndices(Idx);
// 	LRF_est.setRadiusSearch(sup_radius);
// 	LRF_est.compute(*pcl_LRF);

// 	//SHOT
// 	pcl::SHOTEstimation<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot_est;
// 	shot_est.setInputCloud(cloud);
// 	shot_est.setInputNormals(normals);
// 	pcl::PointCloud<pcl::SHOT352>::Ptr shots(new pcl::PointCloud<pcl::SHOT352>());
// 	pcl::search::KdTree<pcl::PointXYZ>::Ptr treeI(new pcl::search::KdTree<pcl::PointXYZ>);
// 	treeI->setInputCloud(cloud);
// 	shot_est.setSearchMethod(tree);
// 	shot_est.setIndices(Idx);
// 	shot_est.setRadiusSearch(sup_radius);
// 	shot_est.compute(*shots);
// 	//
// 	float NORMAL = 1;
// 	for (i = 0; i<Idx->indices.size(); i++)
// 	{
// 		if (abs(pcl_LRF->points[i].x_axis[0]) <= NORMAL)
// 		{
// 			LRF temp;
// 			temp.pointID = Idx->indices[i];
// 			temp.x_axis.x = pcl_LRF->points[i].x_axis[0];
// 			temp.x_axis.y = pcl_LRF->points[i].x_axis[1];
// 			temp.x_axis.z = pcl_LRF->points[i].x_axis[2];
// 			temp.y_axis.x = pcl_LRF->points[i].y_axis[0];
// 			temp.y_axis.y = pcl_LRF->points[i].y_axis[1];
// 			temp.y_axis.z = pcl_LRF->points[i].y_axis[2];
// 			temp.z_axis.x = pcl_LRF->points[i].z_axis[0];
// 			temp.z_axis.y = pcl_LRF->points[i].z_axis[1];
// 			temp.z_axis.z = pcl_LRF->points[i].z_axis[2];
// 			LRFs.push_back(temp);
// 		}
// 		else
// 		{
// 			LRF temp = { NULL_POINTID,{ 1.0f,0.0f,0.0f },{ 0.0f,1.0f,0.0f },{ 0.0f,0.0f,1.0f } };
// 			LRFs.push_back(temp);
// 		}
// 	}
// 	features.resize(shots->points.size());
// 	for (i = 0; i<features.size(); i++)
// 	{
// 		features[i].resize(352);
// 		for (j = 0; j<352; j++)
// 		{
// 			features[i][j] = shots->points[i].descriptor[j];
// 		}
// 	}
// }

void transformCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, LRF pointLRF, pcl::PointCloud<pcl::PointXYZ>::Ptr &transformed_cloud)
{
	pcl::PointXYZ point = cloud->points[0];//the keypoint
	int number_of_points = cloud->points.size();
	transformed_cloud->points.resize(number_of_points);
	Eigen::Matrix3f matrix;
	matrix(0, 0) = pointLRF.x_axis.x; matrix(0, 1) = pointLRF.x_axis.y; matrix(0, 2) = pointLRF.x_axis.z;
	matrix(1, 0) = pointLRF.y_axis.x; matrix(1, 1) = pointLRF.y_axis.y; matrix(1, 2) = pointLRF.y_axis.z;
	matrix(2, 0) = pointLRF.z_axis.x; matrix(2, 1) = pointLRF.z_axis.y; matrix(2, 2) = pointLRF.z_axis.z;
	for (int i = 0; i<number_of_points; i++)
	{
		Eigen::Vector3f transformed_point(
			cloud->points[i].x - point.x,
			cloud->points[i].y - point.y,
			cloud->points[i].z - point.z);

		transformed_point = matrix * transformed_point;

		pcl::PointXYZ new_point;
		new_point.x = transformed_point(0);
		new_point.y = transformed_point(1);
		new_point.z = transformed_point(2);
		transformed_cloud->points[i] = new_point;
	}
}

PointCloudPtr removeInvalidkeyPoint(PointCloudPtr cloud_in, vector<int> &keyPointIdx, PointCloudPtr keyPoint, float resolution)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr final_keyPoint(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	std::vector<int>index;
	std::vector<float> Dist;
	kdtree.setInputCloud(cloud_in);
	vector<int>keyPointTempIdx;
	for (int i = 0; i < keyPoint->size(); i++)
	{
		kdtree.radiusSearch(cloud_in->points[keyPointIdx[i]], 15 * resolution, index, Dist);
		if (index.size() >= 10)
		{
			keyPointTempIdx.push_back(keyPointIdx[i]);
			final_keyPoint->push_back(keyPoint->points[i]);
		}
		index.clear();
		Dist.clear();
	}
	keyPointIdx = keyPointTempIdx;
	return final_keyPoint;
}
pcl::PointCloud<pcl::PointXYZ>::Ptr getHarris3D_detector(PointCloudPtr cloud, float NMS_radius, vector<int>&key_indices)
{
	key_indices.clear();
	pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI> detector;
	detector.setNonMaxSupression(true);
	detector.setRadius(NMS_radius);
	detector.setInputCloud(cloud);
	detector.setRefine(false);
	pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZI>());
	detector.compute(*keypoints);
	//
	pcl::PointCloud<pcl::PointXYZ>::Ptr _keypoints(new pcl::PointCloud<pcl::PointXYZ>());
	for (int i = 0; i<keypoints->points.size(); i++)
	{
		pcl::PointXYZ p;
		p.x = keypoints->points[i].x;
		p.y = keypoints->points[i].y;
		p.z = keypoints->points[i].z;
		_keypoints->points.push_back(p);
	}
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	vector<int> Idx;
	vector<float> Dist;
	kdtree.setInputCloud(cloud);
	for (int i = 0; i<_keypoints->size(); i++)
	{
		kdtree.nearestKSearch(_keypoints->points[i], 1, Idx, Dist);
		key_indices.push_back(Idx[0]);
	}
	return _keypoints;
}
pcl::PointCloud<pcl::PointXYZ>::Ptr getISS3dKeypoint(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, float resolution, vector<int>&key_indices)
{
	pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss_det;
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr keyPoint(new pcl::PointCloud<pcl::PointXYZ>);
	//参数设置
	iss_det.setSearchMethod(tree);
	iss_det.setSalientRadius(2.7 * resolution);//
	iss_det.setNonMaxRadius(1.8 * resolution);//
	iss_det.setThreshold21(0.975);
	iss_det.setThreshold32(0.975);
	iss_det.setMinNeighbors(5);
	iss_det.setNumberOfThreads(4);
	iss_det.setInputCloud(cloud_in);
	iss_det.compute(*keyPoint);
	pcl::PointCloud<pcl::PointXYZ>::Ptr _keypoints(new pcl::PointCloud<pcl::PointXYZ>());
	for (int i = 0; i<keyPoint->points.size(); i++)
	{
		pcl::PointXYZ p;
		p.x = keyPoint->points[i].x;
		p.y = keyPoint->points[i].y;
		p.z = keyPoint->points[i].z;
		_keypoints->points.push_back(p);
	}
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	vector<int> Idx;
	vector<float> Dist;
	kdtree.setInputCloud(cloud_in);
	for (int i = 0; i<_keypoints->size(); i++)
	{
		kdtree.nearestKSearch(_keypoints->points[i], 1, Idx, Dist);
		key_indices.push_back(Idx[0]);
	}
	return _keypoints;
}
void pointLFSH(float r, int bin_num, Vertex &searchPoint, Vertex& n, pcl::PointCloud<pcl::PointXYZ>::Ptr&neigh, pcl::PointCloud<pcl::Normal>::Ptr&sphere_normals,
	vector<float> &histogram)
{
	int i;
	float nx = n.x;
	float ny = n.y;
	float nz = n.z;
	float x0 = searchPoint.x - nx * r;//x0,y0,z0为searchpoint在平面上的投影
	float y0 = searchPoint.y - ny * r;
	float z0 = searchPoint.z - nz * r;
	float plane_D = -(nx*x0 + ny * y0 + nz * z0);
	//int depth_bin_num=bin_num/3;//参数可调
	int depth_bin_num = bin_num / 3;
	float depth_stride = 2 * r / depth_bin_num;
	int depth_bin_id;
	vector<float> depth_histogram(depth_bin_num, 0);
	/***************/
	//int density_bin_num=bin_num/3;
	int density_bin_num = bin_num / 6;
	float density_stride = r / density_bin_num;
	int density_bin_id;
	vector<float> density_histogram(density_bin_num, 0);
	/***************/
	//int angle_bin_num=bin_num/3;
	int angle_bin_num = bin_num / 2;
	float angle_stride = 180.0f / angle_bin_num;
	int angle_bin_id;
	vector<float> angle_histogram(angle_bin_num, 0);
	/***************/
	float a, b, c;
	float temp_depth, temp_radius, temp_angle;
	for (i = 0; i<neigh->points.size(); i++)
	{
		temp_depth = nx * neigh->points[i].x + ny * neigh->points[i].y + nz * neigh->points[i].z + plane_D;
		c = (neigh->points[i].x - searchPoint.x)*(neigh->points[i].x - searchPoint.x) +
			(neigh->points[i].y - searchPoint.y)*(neigh->points[i].y - searchPoint.y) +
			(neigh->points[i].z - searchPoint.z)*(neigh->points[i].z - searchPoint.z);
		b = (neigh->points[i].x - searchPoint.x)*nx + (neigh->points[i].y - searchPoint.y)*ny + (neigh->points[i].z - searchPoint.z)*nz;
		a = sqrt(abs(c - b * b));
		temp_radius = a;
		temp_angle = sphere_normals->points[i].normal_x*nx + sphere_normals->points[i].normal_y*ny + sphere_normals->points[i].normal_z*nz;
		if (temp_angle>1)
			temp_angle = 1;
		if (temp_angle<-1)
			temp_angle = -1;
		temp_angle = acos(temp_angle) / Pi * 180;
		//统计直方图
		if (temp_depth >= 2 * r)//防止浮点数溢出
			depth_bin_id = depth_bin_num;
		if (temp_depth <= 0.0f)
			depth_bin_id = 1;
		else
			depth_bin_id = temp_depth / depth_stride + 1;

		if (temp_radius >= r)
			density_bin_id = density_bin_num;
		else
			density_bin_id = temp_radius / density_stride + 1;

		if (temp_angle >= 180)
			angle_bin_id = angle_bin_num;
		else
			angle_bin_id = temp_angle / angle_stride + 1;
		//
		depth_histogram[depth_bin_id - 1] += 1 / float(neigh->points.size());
		density_histogram[density_bin_id - 1] += 1 / float(neigh->points.size());
		angle_histogram[angle_bin_id - 1] += 1 / float(neigh->points.size());
	}
	copy(density_histogram.begin(), density_histogram.end(), back_inserter(depth_histogram));
	copy(angle_histogram.begin(), angle_histogram.end(), back_inserter(depth_histogram));
	histogram = depth_histogram;
}
void LFSH_compute(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, vector<int> indices, float sup_radius, int bin_num, vector<vector<float>>&Histograms)
{
	int i, j;
	///////////////////////计算法向量/////////////////////////////
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(cloud);
	n.setInputCloud(cloud);
	n.setSearchMethod(tree);
	n.setKSearch(20);
	n.compute(*normals);
	///////////////////计算包围球//////////////////////////////
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	vector<int>pointIdx;
	vector<float>pointDst;
	kdtree.setInputCloud(cloud);
	pcl::PointXYZ query_point;
	Vertex query_p, query_normal;
	//
	for (i = 0; i < indices.size(); i++)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr sphere_neighbor(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::Normal>::Ptr sphere_normals(new pcl::PointCloud<pcl::Normal>);
		vector<float>hist_temp;
		query_point = cloud->points[indices[i]];
		if (kdtree.radiusSearch(query_point, sup_radius, pointIdx, pointDst) > 5)
		{
			for (j = 0; j < pointIdx.size(); j++)
			{
				sphere_neighbor->points.push_back(cloud->points[pointIdx[j]]);
				sphere_normals->points.push_back(normals->points[pointIdx[j]]);
			}
			Vertex LRA = { normals->points[indices[i]].normal_x,normals->points[indices[i]].normal_y,normals->points[indices[i]].normal_z };
			query_p.x = query_point.x;
			query_p.y = query_point.y;
			query_p.z = query_point.z;
			query_normal.x = normals->points[indices[i]].normal_x;
			query_normal.y = normals->points[indices[i]].normal_y;
			query_normal.z = normals->points[indices[i]].normal_z;
			pointLFSH(sup_radius, bin_num, query_p, query_normal, sphere_neighbor, sphere_normals, hist_temp);
			//
			Histograms.push_back(hist_temp);
		}
		else
		{
			vector<float> f_null(bin_num, 0.0f);
			Histograms.push_back(f_null);
		}
	}
}
void LRF_Z_axis(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, Vertex &z_axis)
{
	int i;
	pcl::PointXYZ query_point = cloud->points[0];

	// Covariance matrix
	Eigen::Matrix3f Cov;
	Eigen::Vector4f centroid;
	pcl::compute3DCentroid(*cloud, centroid);
	pcl::computeCovarianceMatrix(*cloud, centroid, Cov);
	EIGEN_ALIGN16 Eigen::Vector3f::Scalar eigen_min;
	EIGEN_ALIGN16 Eigen::Vector3f normal;
	pcl::eigen33(Cov, eigen_min, normal);
	z_axis.x = normal(0);
	z_axis.y = normal(1);
	z_axis.z = normal(2);
	// disambiguity the sign of Z-axis
	float z_sign = 0;
	for (i = 0; i<cloud->points.size(); i++)
	{
		float vec_x = query_point.x - cloud->points[i].x;
		float vec_y = query_point.y - cloud->points[i].y;
		float vec_z = query_point.z - cloud->points[i].z;
		z_sign += (vec_x*z_axis.x + vec_y * z_axis.y + vec_z * z_axis.z);
	}
	if (z_sign<0)
	{
		z_axis.x = -z_axis.x;
		z_axis.y = -z_axis.y;
		z_axis.z = -z_axis.z;
	}
}
void LRF_X_axis(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, Vertex z_axis, float sup_radius, vector<float>  PointDist, Vertex &x_axis)//TOLDI method for x_axis
{
	int i;
	pcl::PointXYZ query_point = cloud->points[0];
	//
	vector<Vertex> vec_proj;
	vector<float>dist_weight, sign_weight;//two weights for each projection vector
	for (i = 0; i<cloud->points.size(); i++)
	{
		Vertex temp;
		Vertex pq = { cloud->points[i].x - query_point.x,cloud->points[i].y - query_point.y,cloud->points[i].z - query_point.z };
		float proj = z_axis.x*pq.x + z_axis.y*pq.y + z_axis.z*pq.z;
		if (proj >= 0)
			sign_weight.push_back(pow(proj, 2));
		else
			sign_weight.push_back(-pow(proj, 2));
		temp.x = pq.x - proj * z_axis.x;
		temp.y = pq.y - proj * z_axis.y;
		temp.z = pq.z - proj * z_axis.z;
		vec_proj.push_back(temp);
	}

	for (i = 0; i<PointDist.size(); i++)
	{
		float wei_temp = sup_radius - sqrt(PointDist[i]);
		wei_temp = pow(wei_temp, 2);
		dist_weight.push_back(wei_temp);
	}
	Vertex x_axis_temp = { 0.0f,0.0f,0.0f };
	for (i = 0; i<cloud->points.size(); i++)
	{
		float weight_sum = dist_weight[i] * sign_weight[i];
		x_axis_temp.x += weight_sum * vec_proj[i].x;
		x_axis_temp.y += weight_sum * vec_proj[i].y;
		x_axis_temp.z += weight_sum * vec_proj[i].z;
	}
	//normalize
	float size = sqrt(pow(x_axis_temp.x, 2) + pow(x_axis_temp.y, 2) + pow(x_axis_temp.z, 2));
	x_axis_temp.x /= size; x_axis_temp.y /= size; x_axis_temp.z /= size;
	x_axis = x_axis_temp;
}
void LRF_axis(Vertex x_axis, Vertex z_axis, Vertex &y_axis)
{
	Eigen::Vector3f x(x_axis.x, x_axis.y, x_axis.z);
	Eigen::Vector3f z(z_axis.x, z_axis.y, z_axis.z);
	Eigen::Vector3f y;
	y = x.cross(z);
	y_axis.x = y(0);
	y_axis.y = y(1);
	y_axis.z = y(2);
}
void RotateCloud(const LRF pointLRF, const float angle, const PointCloudPtr cloud, PointCloudPtr&rotated_cloud)
{
	Eigen::Matrix4f rotation_matrix;
	float sx, sy, sz, cx, cy, cz;
	sx = (angle / 180)*Pi; sy = (angle / 180)*Pi; sz = (angle / 180)*Pi;
	cx = sx; cy = sy; cz = sz;
	sx = sin(sx); sy = sin(sy); sz = sin(sz); cx = cos(cx); cy = cos(cy); cz = cos(cz);
	//
	rotation_matrix(0, 0) = cy * cz; rotation_matrix(0, 1) = cy * sz; rotation_matrix(0, 2) = -sy; rotation_matrix(0, 3) = 0;
	rotation_matrix(1, 0) = sx * sy*cz - cx * sz; rotation_matrix(1, 1) = sx * sy*sz + cx * cz; rotation_matrix(1, 2) = sx * cy; rotation_matrix(1, 3) = 0;
	rotation_matrix(2, 0) = cx * sy*cz + sx * sz; rotation_matrix(2, 1) = cx * sy*sz - sx * cz; rotation_matrix(2, 2) = cx * cy; rotation_matrix(2, 3) = 0;
	rotation_matrix(3, 0) = 0; rotation_matrix(3, 1) = 0; rotation_matrix(3, 2) = 0; rotation_matrix(3, 3) = 1;
	//
	pcl::transformPointCloud(*cloud, *rotated_cloud, rotation_matrix);
}
void Eigen_analysis(pcl::PointCloud<pcl::PointXYZ>::Ptr local_cloud, float &eig1, float &eig2, float &eig3)//从小到大排序
{
	Eigen::Matrix3f Cov;
	Eigen::Vector4f centroid;
	pcl::compute3DCentroid(*local_cloud, centroid);
	pcl::computeCovarianceMatrix(*local_cloud, centroid, Cov);
	Eigen::EigenSolver <Eigen::Matrix3f> eigen_solver;
	eigen_solver.compute(Cov);

	unsigned int temp = 0;
	unsigned int major_index = 0;
	unsigned int middle_index = 1;
	unsigned int minor_index = 2;
	Eigen::Vector3f major_axis, middle_axis, minor_axis;
	Eigen::EigenSolver <Eigen::Matrix3f>::EigenvectorsType eigen_vectors;
	Eigen::EigenSolver <Eigen::Matrix3f>::EigenvalueType eigen_values;
	eigen_vectors = eigen_solver.eigenvectors();
	eigen_values = eigen_solver.eigenvalues();
	if (eigen_values.real() (major_index) < eigen_values.real() (middle_index))
	{
		temp = major_index;
		major_index = middle_index;
		middle_index = temp;
	}
	if (eigen_values.real() (major_index) < eigen_values.real() (minor_index))
	{
		temp = major_index;
		major_index = minor_index;
		minor_index = temp;
	}
	if (eigen_values.real() (middle_index) < eigen_values.real() (minor_index))
	{
		temp = minor_index;
		minor_index = middle_index;
		middle_index = temp;
	}
	eig1 = eigen_values.real() (minor_index);
	eig2 = eigen_values.real() (middle_index);
	eig3 = eigen_values.real() (major_index);
}
bool RCS_SortBydist(const Vertex_d_ang v1, const Vertex_d_ang v2)
{
	return v1.dist<v2.dist;//ascending
}
void RCS_per_rotate(PointCloudPtr rotated_cloud, int num_of_contour_points, float sup_radius, vector<float>&signature)
{
	int i;
	vector<Vertex_d_ang> rotated_points;
	//compute the attribute per point
	for (i = 0; i<rotated_cloud->points.size(); i++)
	{
		float dist = sqrt(rotated_cloud->points[i].x*rotated_cloud->points[i].x + rotated_cloud->points[i].y*rotated_cloud->points[i].y);
		float ang;
		float abs_sin;
		if (dist == 0)
			abs_sin = 0;
		else
			abs_sin = abs(rotated_cloud->points[i].x) / dist;
		if (abs_sin>1) abs_sin = 1;
		if (abs_sin + 1<0) abs_sin = -0.9999;//avoid sin errors
		float abs_ang = asin(abs_sin) * 180 / Pi;
		if ((rotated_cloud->points[i].x >= 0) && (rotated_cloud->points[i].y >= 0)) ang = abs_ang;//1 quadrant
		else if ((rotated_cloud->points[i].x >= 0) && (rotated_cloud->points[i].y<0)) ang = 360 - abs_ang;//4 quadrant
		else if ((rotated_cloud->points[i].x<0) && (rotated_cloud->points[i].y<0)) ang = 180 + abs_ang;//3 quadrant
		else ang = 180 - abs_ang;//2 quadrant
		Vertex_d_ang temp = { rotated_cloud->points[i].x,rotated_cloud->points[i].y,rotated_cloud->points[i].z,dist,ang };
		rotated_points.push_back(temp);
	}
	//compute contour points
	float ang_step = 360 / num_of_contour_points;
	float ang_thresh = 15;//this thresh is used to find the point set in the projected point map for approximately determing a contour point,  degrees
	vector<vector<Vertex_d_ang>> point_clusters;
	point_clusters.resize(num_of_contour_points);
	for (i = 0; i<rotated_points.size(); i++)
	{
		int mod = int(rotated_points[i].angle_to_axis) % int(ang_step);
		int mul = int(rotated_points[i].angle_to_axis) / int(ang_step);
		if ((mod <= ang_thresh) && (mul<num_of_contour_points))
			point_clusters[mul].push_back(rotated_points[i]);
	}
	//compute coutour signature
	signature.resize(num_of_contour_points);
	for (i = 0; i<point_clusters.size(); i++)
	{
		if (point_clusters[i].size() == 0)
			signature[i] = 0;
		else
		{
			sort(point_clusters[i].begin(), point_clusters[i].end(), RCS_SortBydist);
			int opt_id = point_clusters[i].size() - 1;
			signature[i] = (point_clusters[i][opt_id].dist) / sup_radius;//nomalization
		}
	}
}
void RCS_compute(pcl::PointCloud<pcl::PointXYZ>::Ptr  cloud, vector<int> indices, float sup_radius, float rotate_angle, int num_of_rotations,
	int num_of_contour_points, vector<vector<float>>&Histograms)
{
	int i, j;
	//local surface determination via KD-tree
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	vector<int>pointIdx;
	vector<float>pointDst;
	kdtree.setInputCloud(cloud);
	pcl::PointXYZ query_point;

	for (i = 0; i<indices.size(); i++)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr sphere_neighbor(new pcl::PointCloud<pcl::PointXYZ>);//local surface
		pcl::PointCloud<pcl::PointXYZ>::Ptr sphere_neighbor_z(new pcl::PointCloud<pcl::PointXYZ>);//subset of the local surface for computing the z-axis of the LRF
		pcl::PointCloud<pcl::PointXYZ>::Ptr sphere_neighbor_trans(new pcl::PointCloud<pcl::PointXYZ>);//local surface in the LRF system, invariant to rotation
		Vertex x_axis, y_axis, z_axis;
		query_point = cloud->points[indices[i]];
		//
		if (kdtree.radiusSearch(query_point, sup_radius / 2, pointIdx, pointDst)>3)//if the point count in the sphere_neighbor_z is smaller than 4, discard computing RCS
		{
			for (j = 0; j<pointIdx.size(); j++)
			{
				sphere_neighbor_z->points.push_back(cloud->points[pointIdx[j]]);
			}
			LRF_Z_axis(sphere_neighbor_z, z_axis);
		}
		else
		{
			vector<float> RCS_feature_NULL(num_of_rotations*num_of_contour_points, 0.0f);
			Histograms.push_back(RCS_feature_NULL);
			continue;
		}
		if (kdtree.radiusSearch(query_point, sup_radius, pointIdx, pointDst)>10)//if the point count in the local surface is smaller than 10, discard computing RCS
		{
			for (j = 0; j<pointIdx.size(); j++)
			{
				sphere_neighbor->points.push_back(cloud->points[pointIdx[j]]);
			}
			LRF_X_axis(sphere_neighbor, z_axis, sup_radius, pointDst, x_axis);
			LRF_axis(x_axis, z_axis, y_axis);
			LRF pointLRF = { indices[i],x_axis,y_axis,z_axis };//LRF calculation 
			transformCloud(sphere_neighbor, pointLRF, sphere_neighbor_trans);//transform the local surface to the LRF system
			vector<float> RCS_feature;
			for (int r = 0; r<num_of_rotations; r++)
			{
				float rotate_angle_temp = r * rotate_angle;
				pcl::PointCloud<pcl::PointXYZ>::Ptr sphere_neighbor_trans_rotated(new pcl::PointCloud<pcl::PointXYZ>);//rotated surface
				RotateCloud(pointLRF, rotate_angle_temp, sphere_neighbor_trans, sphere_neighbor_trans_rotated);
				vector<float> signature;
				RCS_per_rotate(sphere_neighbor_trans_rotated, num_of_contour_points, sup_radius, signature);
				std::copy(signature.begin(), signature.end(), std::back_inserter(RCS_feature));
			}
			Histograms.push_back(RCS_feature);
		}
		else
		{
			vector<float> RCS_feature_NULL(num_of_rotations*num_of_contour_points, 0.0f);
			Histograms.push_back(RCS_feature_NULL);
		}
	}
}
bool SortBySaliency(const ISS_Key_Type&v1, const ISS_Key_Type&v2)
{
	return v1.saliency<v2.saliency;//升序排列
}
void ISS_Non_Max_Sup(PointCloudPtr cloud, vector<ISS_Key_Type>&pnts, float NMS_radius)
{
	int i, j;
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	vector<int>pointIdx;
	vector<float>pointDst;
	kdtree.setInputCloud(cloud);
	for (i = 0; i<cloud->points.size(); i++)
	{
		if (pnts[i].TorF == true)
		{
			kdtree.radiusSearch(cloud->points[i], NMS_radius, pointIdx, pointDst);
			vector<ISS_Key_Type> local_key_pnts;
			for (j = 0; j<pointIdx.size(); j++)
			{
				local_key_pnts.push_back(pnts[pointIdx[j]]);
			}
			vector<ISS_Key_Type> local_copy = local_key_pnts;
			sort(local_copy.begin(), local_copy.end(), SortBySaliency);
			//float max_local_saliency=local_copy[local_copy.size()-1].saliency;
			if (pnts[i].PointID != local_copy[local_copy.size() - 1].PointID)
				pnts[i].TorF = false;
		}

	}
}

void ISS_detector(PointCloudPtr cloud, float mr, float support_radius, vector<int>&key_indices)
{
	int i, j;
	float th12 = 0.975, th23 = 0.975;
	key_indices.resize(0);//注意初始化
						  //计算当前搜寻半径下的局部曲面大小
	vector<ISS_Key_Type> pnts;
	pnts.resize(cloud->points.size());//pnts include all the points with tags
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	vector<int>pointIdx;
	vector<float>pointDst;
	kdtree.setInputCloud(cloud);
	pcl::PointXYZ query_point;
	for (i = 0; i<cloud->points.size(); i++)
	{
		ISS_Key_Type temp;
		pcl::PointCloud<pcl::PointXYZ>::Ptr sphere_neighbor(new pcl::PointCloud<pcl::PointXYZ>);//初始局部曲面
		query_point = cloud->points[i];
		if (kdtree.radiusSearch(query_point, support_radius, pointIdx, pointDst) <= 3)
		{
			ISS_Key_Type a = { i,0,0,NULL_Saliency,false };
			temp = a;
		}
		else
		{
			for (j = 0; j<pointIdx.size(); j++)
			{
				sphere_neighbor->points.push_back(cloud->points[pointIdx[j]]);//确定局部曲面
			}
			float eig1, eig2, eig3;
			Eigen_analysis(sphere_neighbor, eig1, eig2, eig3);
			temp.PointID = i;
			temp.eig1_2 = eig1 / eig2;
			temp.eig2_3 = eig2 / eig3;
			temp.saliency = eig1;
			if ((temp.eig1_2<th12) && (temp.eig2_3<th23))
				temp.TorF = true;
			else
			{
				temp.TorF = false;
				temp.saliency = NULL_Saliency;
			}
		}
		pnts[i] = temp;
	}
	//float NMS_radius=3*MeshResolution_mr_compute(cloud);//NMSradius 应和当前mr先关，与sup_radius不同
	float NMS_radius = 2.5*mr;
	ISS_Non_Max_Sup(cloud, pnts, NMS_radius);
	for (i = 0; i<pnts.size(); i++)
	{
		if (pnts[i].TorF == true)
			key_indices.push_back(i);
	}
	//printf("No. of ISS detected points:%d\n",key_indices.size());
}
void Harris3D_detector(PointCloudPtr cloud, float NMS_radius, vector<int>&key_indices)
{
	key_indices.clear();
	pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI> detector;
	detector.setNonMaxSupression(true);
	detector.setRadius(NMS_radius);
	detector.setInputCloud(cloud);
	detector.setRefine(false);
	pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZI>());
	detector.compute(*keypoints);
	//
	pcl::PointCloud<pcl::PointXYZ>::Ptr _keypoints(new pcl::PointCloud<pcl::PointXYZ>());
	for (int i = 0; i<keypoints->points.size(); i++)
	{
		pcl::PointXYZ p;
		p.x = keypoints->points[i].x;
		p.y = keypoints->points[i].y;
		p.z = keypoints->points[i].z;
		_keypoints->points.push_back(p);
	}
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	vector<int> Idx;
	vector<float> Dist;
	kdtree.setInputCloud(cloud);
	for (int i = 0; i<_keypoints->size(); i++)
	{
		kdtree.nearestKSearch(_keypoints->points[i], 1, Idx, Dist);
		key_indices.push_back(Idx[0]);
	}
}
// void FPFH_descriptor(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, std::vector<int>& indices,
//                      float sup_radius, std::vector<std::vector<float>>& features){
//     int i, j;
//     //pcl::PointIndicesPtr Idx = std::make_shared<pcl::PointIndices>();
//     pcl::PointIndicesPtr Idx = boost::shared_ptr<pcl::PointIndices>();
//     for (j = 0; j < indices.size(); j++)
//         Idx->indices.push_back(indices[j]);
//     pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> n;
//     pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
//     pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
//     tree->setInputCloud(cloud);
//     n.setInputCloud(cloud);
//     n.setSearchMethod(tree);
//     n.setRadiusSearch(sup_radius / 4);
//     n.compute(*normals);
//     // Create the FPFH estimation class, and pass the input dataset+normals to it
//     pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
//     fpfh.setInputCloud (cloud);
//     fpfh.setInputNormals (normals);
//     // alternatively, if cloud is of tpe PointNormal, do fpfh.setInputNormals (cloud);

//     // Create an empty kdtree representation, and pass it to the FPFH estimation object.
//     // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).

//     fpfh.setSearchMethod (tree);
//     fpfh.setIndices(Idx);
//     // Output datasets
//     pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs (new pcl::PointCloud<pcl::FPFHSignature33> ());

//     // Use all neighbors in a sphere of radius 5cm
//     // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
//     fpfh.setRadiusSearch (sup_radius);

//     // Compute the features
//     fpfh.compute (*fpfhs);
//     features.resize(fpfhs->points.size());
//     for(i = 0; i < features.size(); i++){
//         features[i].resize(33);
//         for(j = 0; j < 33; j++){
//             features[i][j] = fpfhs->points[i].histogram[j];
//         }
//     }
// }
void FPFH_descriptor(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float sup_radius, std::vector<std::vector<float>>& features){
    int i, j;
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> n;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);
    n.setInputCloud(cloud);
    n.setSearchMethod(tree);
    n.setRadiusSearch(sup_radius / 4);
    n.compute(*normals);
    // Create the FPFH estimation class, and pass the input dataset+normals to it
    pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
    fpfh.setInputCloud (cloud);
    fpfh.setInputNormals (normals);
    // alternatively, if cloud is of tpe PointNormal, do fpfh.setInputNormals (cloud);

    // Create an empty kdtree representation, and pass it to the FPFH estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    fpfh.setSearchMethod (tree);
    // Output datasets
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs (new pcl::PointCloud<pcl::FPFHSignature33> ());

    // Use all neighbors in a sphere of radius 5cm
    // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
    fpfh.setRadiusSearch (sup_radius);

    // Compute the features
    fpfh.compute (*fpfhs);
    features.resize(fpfhs->points.size());
    for(i = 0; i < features.size(); i++){
        features[i].resize(33);
        for(j = 0; j < 33; j++){
            features[i][j] = fpfhs->points[i].histogram[j];
        }
    }
}
vector<int> removeInvalidPoint(PointCloudPtr cloud_in, vector<int> &keyPointIdx, float resolution)
{
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	std::vector<int>index;
	std::vector<float> Dist;
	kdtree.setInputCloud(cloud_in);
	vector<int>keyPointTempIdx;
	for (int i = 0; i < keyPointIdx.size(); i++)
	{
		kdtree.radiusSearch(cloud_in->points[keyPointIdx[i]], 15 * resolution, index, Dist);
		if (index.size() >= 10)
		{
			keyPointTempIdx.push_back(keyPointIdx[i]);
		}
		index.clear();
		Dist.clear();
	}
	return keyPointTempIdx;
}
