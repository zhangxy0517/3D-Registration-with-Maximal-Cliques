#include <stdio.h>
#include <vector>
#include <time.h>
#include <algorithm>
#include <pcl/point_types.h>
#include <pcl/registration/transforms.h>
#define BOOST_TYPEOF_EMULATION
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/shot.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/mls.h>
#include <pcl/visualization/pcl_visualizer.h>
#include "Eva.h"
/*******************************************************************************dataload********************************************************/
int XYZorMeshlabPly_Read(string Filename, PointCloudPtr& cloud)
{
	int i;
	int nXYZ_nums;
	vector<Vertex> vXYZ;
	FILE* fp = fopen(Filename.c_str(), "r");
	if (fp == NULL)
	{
		printf("File can't open!\n");
		return -1;
	}
	const char* FILEPATH = Filename.c_str();
	char a = FILEPATH[strlen(FILEPATH) - 1];
	//
	if (a == 'y')
	{
		char str[1024];
		fscanf(fp, "%s\n", &str);
		fscanf(fp, "%s %s %s\n", &str, &str, &str);
		fscanf(fp, "%s %s %s\n", &str, &str, &str);
		fscanf(fp, "%s %s %d\n", &str, &str, &nXYZ_nums);
		fscanf(fp, "%s %s %s\n", &str, &str, &str);
		fscanf(fp, "%s %s %s\n", &str, &str, &str);
		fscanf(fp, "%s %s %s\n", &str, &str, &str);
		fscanf(fp, "%s %s %s\n", &str, &str, &str);
		fscanf(fp, "%s %s %s\n", &str, &str, &str);
		fscanf(fp, "%s %s %s\n", &str, &str, &str);
		fscanf(fp, "%s %s %s\n", &str, &str, &str);
		fscanf(fp, "%s %s %s\n", &str, &str, &str);
		fscanf(fp, "%s %s %s %s %s\n", &str, &str, &str, &str, &str);
		fscanf(fp, "%s %s %s\n", &str, &str, &str);
		fscanf(fp, "%s\n", &str);
	}
	else
	{
		fscanf(fp, "%d\n", &nXYZ_nums);
	}
	vXYZ.resize(nXYZ_nums);
	if (a == 'y')
	{
		float x, y, z, intens;
		for (i = 0; i < vXYZ.size(); i++)
		{
			fscanf(fp, "%f %f %f %f %f %f %f\n", &vXYZ[i].x, &vXYZ[i].y, &vXYZ[i].z, &x, &y, &z, &intens);
		}
	}
	else
	{
		for (i = 0; i < vXYZ.size(); i++)
		{
			fscanf(fp, "%f %f %f\n", &vXYZ[i].x, &vXYZ[i].y, &vXYZ[i].z);
		}
	}

	fclose(fp);
	cloud->width = vXYZ.size();
	cloud->height = 1;
	cloud->is_dense = true;
	cloud->points.resize(cloud->width * cloud->height);
	for (i = 0; i < cloud->points.size(); i++)
	{
		cloud->points[i].x = vXYZ[i].x;
		cloud->points[i].y = vXYZ[i].y;
		cloud->points[i].z = vXYZ[i].z;
	}
	return 0;
}
int XYZorPly_Read(string Filename, PointCloudPtr& cloud)
{
	int i;
	int nXYZ_nums;
	vector<Vertex> vXYZ;
	FILE* fp = fopen(Filename.c_str(), "r");
	if (fp == NULL)
	{
		printf("File can't open!\n");
		return -1;
	}
	const char* FILEPATH = Filename.c_str();
	char a = FILEPATH[strlen(FILEPATH) - 1];
	//
	if (a == 'y')
	{
		char str[1024];
		fscanf(fp, "%s\n", &str);
		fscanf(fp, "%s %s %s\n", &str, &str, &str);
		fscanf(fp, "%s %s %d\n", &str, &str, &nXYZ_nums);
		fscanf(fp, "%s %s %s\n", &str, &str, &str);
		fscanf(fp, "%s %s %s\n", &str, &str, &str);
		fscanf(fp, "%s %s %s\n", &str, &str, &str);
		fscanf(fp, "%s %s %s\n", &str, &str, &str);
		fscanf(fp, "%s %s %s %s %s\n", &str, &str, &str, &str, &str);
		fscanf(fp, "%s\n", &str);
	}
	else
	{
		fscanf(fp, "%d\n", &nXYZ_nums);
	}
	vXYZ.resize(nXYZ_nums);
	for (i = 0; i < vXYZ.size(); i++)
	{
		fscanf(fp, "%f %f %f\n", &vXYZ[i].x, &vXYZ[i].y, &vXYZ[i].z);
	}
	fclose(fp);
	cloud->width = vXYZ.size();
	cloud->height = 1;
	cloud->is_dense = true;
	cloud->points.resize(cloud->width * cloud->height);
	for (i = 0; i < cloud->points.size(); i++)
	{
		cloud->points[i].x = vXYZ[i].x;
		cloud->points[i].y = vXYZ[i].y;
		cloud->points[i].z = vXYZ[i].z;
	}
	return 0;
}

void write_cloud(PointCloudPtr cloud, string file_name)
{
	FILE* fpo = fopen(file_name.c_str(), "w");
	fprintf(fpo, "%d\n", cloud->points.size());
	for (int i = 0; i < cloud->points.size(); i++)
	{
		fprintf(fpo, "%f %f %f\n", cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
	}
	fclose(fpo);
}
void Cloud2Meshlab_showfile(string file_open, string file_save)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	XYZorMeshlabPly_Read(file_open, cloud);
	float Triangle_radius = 10 * MeshResolution_mr_compute(cloud);
	//
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(cloud);
	n.setInputCloud(cloud);
	n.setSearchMethod(tree);
	n.setKSearch(20);
	n.compute(*normals);
	//
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
	pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);
	pcl::search::KdTree<pcl::PointNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointNormal>);
	tree2->setInputCloud(cloud_with_normals);
	// Initialize objects2014/1/22 15:32:35
	pcl::PolygonMesh triangles;
	pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
	gp3.setSearchRadius(Triangle_radius);
	gp3.setMu(2.5);
	gp3.setMaximumNearestNeighbors(100);
	gp3.setMaximumSurfaceAngle(M_PI / 4); // 45 degrees
	gp3.setMinimumAngle(M_PI / 6); // 30 degrees
	gp3.setMaximumAngle(2 * M_PI / 3); // 120 degrees
	gp3.setNormalConsistency(false);
	// Get result
	gp3.setInputCloud(cloud_with_normals);
	gp3.setSearchMethod(tree2);
	gp3.reconstruct(triangles);
	FILE* fp = fopen(file_save.c_str(), "w");
	fprintf(fp, "ply\n");
	fprintf(fp, "format ascii 1.0\n");
	fprintf(fp, "%s %s %d\n", "element", "vertex", cloud->points.size());
	fprintf(fp, "property float x\n");
	fprintf(fp, "property float y\n");
	fprintf(fp, "property float z\n");
	fprintf(fp, "%s %s %d\n", "element", "face", triangles.polygons.size());
	fprintf(fp, "property list uchar int vertex_index\n");
	fprintf(fp, "end_header\n");
	for (int i = 0; i < cloud->points.size(); i++)
		fprintf(fp, "%f %f %f\n", cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
	for (int i = 0; i < triangles.polygons.size(); i++)
		fprintf(fp, "%d %d %d %d\n", 3, triangles.polygons[i].vertices[0], triangles.polygons[i].vertices[1], triangles.polygons[i].vertices[2]);
	fclose(fp);
}
float MeshResolution_mr_compute(PointCloudPtr& cloud)
{
	int i;
	//计算点云分辨率
	float mr = 0;
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	vector<int>pointIdx;
	vector<float>pointDst;
	kdtree.setInputCloud(cloud);
	pcl::PointXYZ query_point;
	for (i = 0; i < cloud->points.size(); i++)
	{
		query_point = cloud->points[i];
		kdtree.nearestKSearch(query_point, 2, pointIdx, pointDst);
		float x = cloud->points[pointIdx[0]].x - cloud->points[pointIdx[1]].x;
		float y = cloud->points[pointIdx[0]].y - cloud->points[pointIdx[1]].y;
		float z = cloud->points[pointIdx[0]].z - cloud->points[pointIdx[1]].z;
		float mr_temp = sqrt(x * x + y * y + z * z);
		mr += mr_temp;
	}
	mr /= cloud->points.size();
	return mr;//approximate calculation
}
//int GTMatRead(string &Filename, Eigen::Matrix4d& Mat_GT)
//{
//	FILE* fp = fopen(Filename.c_str(), "r");
//	if (fp == NULL)
//	{
//		printf("Mat File can't open!\n");
//		return -1;
//	}
//	fscanf(fp, "%lf %lf %lf %lf\n", &Mat_GT(0, 0), &Mat_GT(0, 1), &Mat_GT(0, 2), &Mat_GT(0, 3));
//	fscanf(fp, "%lf %lf %lf %lf\n", &Mat_GT(1, 0), &Mat_GT(1, 1), &Mat_GT(1, 2), &Mat_GT(1, 3));
//	fscanf(fp, "%lf %lf %lf %lf\n", &Mat_GT(2, 0), &Mat_GT(2, 1), &Mat_GT(2, 2), &Mat_GT(2, 3));
//	fscanf(fp, "%lf %lf %lf %lf\n", &Mat_GT(3, 0), &Mat_GT(3, 1), &Mat_GT(3, 2), &Mat_GT(3, 3));
//	fclose(fp);
//}

int Voxel_grid_downsample(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& new_cloud,
                      float leaf_size) {
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(leaf_size, leaf_size, leaf_size);
    sor.filter(*new_cloud);
    return 0;
}
/*******************************************************************************Feature match********************************************************/
void feature_matching(PointCloudPtr& cloud_source, PointCloudPtr& cloud_target,
                      vector<vector<float>>& feature_source, vector<vector<float>>& feature_target, vector<Corre_3DMatch>& Corres)
{
    int i, j;
    pcl::PointCloud<pcl::SHOT352>::Ptr Feature_source(new pcl::PointCloud<pcl::SHOT352>);
    pcl::PointCloud<pcl::SHOT352>::Ptr Feature_target(new pcl::PointCloud<pcl::SHOT352>);
    Feature_source->points.resize(feature_source.size());
    Feature_target->points.resize(feature_target.size());
    for (i = 0; i < feature_source.size(); i++)
    {
        for (j = 0; j < 352; j++)
        {
            if (j < feature_source[i].size()) Feature_source->points[i].descriptor[j] = feature_source[i][j];
            else Feature_source->points[i].descriptor[j] = 0;
        }
    }
    for (i = 0; i < feature_target.size(); i++)
    {
        for (j = 0; j < 352; j++)
        {
            if (j < feature_target[i].size()) Feature_target->points[i].descriptor[j] = feature_target[i][j];
            else Feature_target->points[i].descriptor[j] = 0;
        }
    }
    //
    pcl::KdTreeFLANN<pcl::SHOT352> kdtree;
    vector<int>Idx;
    vector<float>Dist;
    kdtree.setInputCloud(Feature_target);
    for (i = 0; i < Feature_source->points.size(); i++)
    {
        kdtree.nearestKSearch(Feature_source->points[i], 1, Idx, Dist);
        Corre_3DMatch temp;
        temp.src_index = i;
        temp.des_index = Idx[0];
        temp.src = cloud_source->points[i];
        temp.des = cloud_target->points[Idx[0]];
        temp.score = 1 - sqrt(Dist[0]);
        Corres.push_back(temp);
    }
}
void feature_matching(PointCloudPtr& cloud_source, PointCloudPtr& cloud_target, vector<LRF>LRFs_source, vector<LRF>LRFs_target,
	vector<int>& Idx_source, vector<int>& Idx_target, vector<vector<float>>& feature_source, vector<vector<float>>& feature_target, vector<Corre>& Corres)
{
	int i, j;
	pcl::PointCloud<pcl::SHOT352>::Ptr Feature_source(new pcl::PointCloud<pcl::SHOT352>);
	pcl::PointCloud<pcl::SHOT352>::Ptr Feature_target(new pcl::PointCloud<pcl::SHOT352>);
	Feature_source->points.resize(feature_source.size());
	Feature_target->points.resize(feature_target.size());
	for (i = 0; i < feature_source.size(); i++)
	{
		for (j = 0; j < 352; j++)
		{
			if (j < feature_source[i].size()) Feature_source->points[i].descriptor[j] = feature_source[i][j];
			else Feature_source->points[i].descriptor[j] = 0;
		}
	}
	for (i = 0; i < feature_target.size(); i++)
	{
		for (j = 0; j < 352; j++)
		{
			if (j < feature_target[i].size()) Feature_target->points[i].descriptor[j] = feature_target[i][j];
			else Feature_target->points[i].descriptor[j] = 0;
		}
	}
	//
	pcl::KdTreeFLANN<pcl::SHOT352> kdtree;
	vector<int>Idx;
	vector<float>Dist;
	kdtree.setInputCloud(Feature_target);
	for (i = 0; i < Feature_source->points.size(); i++)
	{
		kdtree.nearestKSearch(Feature_source->points[i], 1, Idx, Dist);
		Corre temp;
		temp.source_idx = Idx_source[i];
		temp.target_idx = Idx_target[Idx[0]];
		temp.source_LRF = LRFs_source[i];
		temp.target_LRF = LRFs_target[Idx[0]];
		temp.score = 0;
		Corres.push_back(temp);
	}
}
void feature_matching_ratio(PointCloudPtr& cloud_source, PointCloudPtr& cloud_target, vector<LRF>LRFs_source, vector<LRF>LRFs_target,
	vector<int>& Idx_source, vector<int>& Idx_target, vector<vector<float>>& feature_source, vector<vector<float>>& feature_target, vector<Corre>& Corres)
{
	int i, j;
	pcl::PointCloud<pcl::SHOT352>::Ptr Feature_source(new pcl::PointCloud<pcl::SHOT352>);
	pcl::PointCloud<pcl::SHOT352>::Ptr Feature_target(new pcl::PointCloud<pcl::SHOT352>);
	Feature_source->points.resize(feature_source.size());
	Feature_target->points.resize(feature_target.size());
	for (i = 0; i < feature_source.size(); i++)
	{
		for (j = 0; j < 352; j++)
		{
			if (j < feature_source[i].size()) Feature_source->points[i].descriptor[j] = feature_source[i][j];
			else Feature_source->points[i].descriptor[j] = 0;
		}
	}
	for (i = 0; i < feature_target.size(); i++)
	{
		for (j = 0; j < 352; j++)
		{
			if (j < feature_target[i].size()) Feature_target->points[i].descriptor[j] = feature_target[i][j];
			else Feature_target->points[i].descriptor[j] = 0;
		}
	}
	//
	pcl::KdTreeFLANN<pcl::SHOT352> kdtree;
	vector<int>Idx;
	vector<float>Dist;
	kdtree.setInputCloud(Feature_target);
	for (i = 0; i < Feature_source->points.size(); i++)
	{
		kdtree.nearestKSearch(Feature_source->points[i], 2, Idx, Dist);
		Corre temp;
		temp.source_idx = Idx_source[i];
		temp.target_idx = Idx_target[Idx[0]];
		temp.source_LRF = LRFs_source[i];
		temp.target_LRF = LRFs_target[Idx[0]];
		if (Dist[1] <= 0.0000001)
			temp.score = 0;
		else
			temp.score = 1 - sqrt(Dist[0]) / sqrt(Dist[1]);
		Corres.push_back(temp);
	}
}

//
void MyType2Eigen(TransMat& M, Eigen::Matrix4f& EigenM)
{
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
			EigenM(i, j) = M.M[i][j];
	}
}
void Add_Gaussian_noise(float dev, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_noise)
{
	boost::mt19937 rng; rng.seed(static_cast<unsigned int> (time(0)));
	boost::normal_distribution<> nd(0, dev);
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor(rng, nd);
	cloud_noise->points.resize(cloud->points.size());
	cloud_noise->header = cloud->header;
	cloud_noise->width = cloud->width;
	cloud_noise->height = cloud->height;

	for (size_t point_i = 0; point_i < cloud->points.size(); ++point_i)
	{
		cloud_noise->points[point_i].x = cloud->points[point_i].x + static_cast<float> (var_nor());
		cloud_noise->points[point_i].y = cloud->points[point_i].y + static_cast<float> (var_nor());
		cloud_noise->points[point_i].z = cloud->points[point_i].z + static_cast<float> (var_nor());
	}
}
void cloud_simp(float size, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_filtered)
{
	pcl::VoxelGrid<pcl::PointXYZ>sor;
	sor.setInputCloud(cloud);
	sor.setLeafSize(size, size, size);
	sor.filter(*cloud_filtered);
}
int Correct_corre_compute(PointCloudPtr cloud_s, PointCloudPtr cloud_t, vector<Corre> Corres, float correct_thresh, Eigen::Matrix4d& GT_mat, string path)
{
	if (Corres.size() == 0) return 0;
	int i;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_s_trans(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*cloud_s, *cloud_s_trans, GT_mat);
	string TC_path = path + "/true_corre.txt";
	FILE* fp = fopen(TC_path.c_str(), "w");
	int Corret_num = 0;
	for (i = 0; i < Corres.size(); i++)
	{
		int Idx_s = Corres[i].source_idx;
		int Idx_t = Corres[i].target_idx;
		float dist = pow(cloud_s_trans->points[Idx_s].x - cloud_t->points[Idx_t].x, 2) + pow(cloud_s_trans->points[Idx_s].y - cloud_t->points[Idx_t].y, 2)
			+ pow(cloud_s_trans->points[Idx_s].z - cloud_t->points[Idx_t].z, 2);
		dist = sqrt(dist);
		if (dist <= correct_thresh) {
			fprintf(fp, "1\n");
			Corret_num++;
		}
		else {
			fprintf(fp, "0\n");
		}
	}
	fclose(fp);
	return Corret_num;
}
void Correct_corre_select(PointCloudPtr cloud_s, PointCloudPtr cloud_t, vector<Corre> Corres, float correct_thresh,
	Eigen::Matrix4f& GT_mat, vector<Corre>& Corres_selected)
{
	int i;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_s_trans(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*cloud_s, *cloud_s_trans, GT_mat);
	//
	int Corret_num = 0;
	for (i = 0; i < Corres.size(); i++)
	{
		int Idx_s = Corres[i].source_idx;
		int Idx_t = Corres[i].target_idx;
		float dist = pow(cloud_s_trans->points[Idx_s].x - cloud_t->points[Idx_t].x, 2) + pow(cloud_s_trans->points[Idx_s].y - cloud_t->points[Idx_t].y, 2)
			+ pow(cloud_s_trans->points[Idx_s].z - cloud_t->points[Idx_t].z, 2);
		dist = sqrt(dist);
		if (dist <= correct_thresh)
			Corres_selected.push_back(Corres[i]);
	}
}
void affinity_matrix_compute(PointCloudPtr cloud_source, PointCloudPtr cloud_target, float mr, vector<Corre> Corres, Eigen::MatrixXf& M)
{
	int i, j;
	//build M matrix
	M = Eigen::MatrixXf::Zero(Corres.size(), Corres.size());
	vector<float>m;
	for (i = 0; i < Corres.size(); i++)
	{
		float affinity_temp = 0.0;
		for (j = i; j < Corres.size(); j++)
		{
			if (j == i)
				affinity_temp = Corres[i].score;
			else
			{
				float x1 = cloud_source->points[Corres[i].source_idx].x - cloud_source->points[Corres[j].source_idx].x;
				float y1 = cloud_source->points[Corres[i].source_idx].y - cloud_source->points[Corres[j].source_idx].y;
				float z1 = cloud_source->points[Corres[i].source_idx].z - cloud_source->points[Corres[j].source_idx].z;
				float x2 = cloud_target->points[Corres[i].target_idx].x - cloud_target->points[Corres[j].target_idx].x;
				float y2 = cloud_target->points[Corres[i].target_idx].y - cloud_target->points[Corres[j].target_idx].y;
				float z2 = cloud_target->points[Corres[i].target_idx].z - cloud_target->points[Corres[j].target_idx].z;
				float a = sqrt(pow(x1, 2) + pow(y1, 2) + pow(z1, 2));
				float b = sqrt(pow(x2, 2) + pow(y2, 2) + pow(z2, 2));
				if ((a != 0.0) && (b != 0.0))
				{
					affinity_temp = a / b;
					if (affinity_temp > b / a) affinity_temp = b / a;
				}
			}
			M(i, j) = affinity_temp;
		}
	}
	for (i = 1; i < Corres.size(); i++)
	{
		for (j = 0; j < i; j++)
			M(i, j) = M(j, i);
	}
}
void find_inlier_corre_id(PointCloudPtr cloud_s, PointCloudPtr cloud_t, vector<Corre> Corres, float correct_thresh, Eigen::Matrix4f& GT_mat, vector<int>& ids)
{
	int i;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_s_trans(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*cloud_s, *cloud_s_trans, GT_mat);
	//
	int Corret_num = 0;
	ids.resize(Corres.size());
	for (i = 0; i < Corres.size(); i++)
	{
		int Idx_s = Corres[i].source_idx;
		int Idx_t = Corres[i].target_idx;
		float dist = pow(cloud_s_trans->points[Idx_s].x - cloud_t->points[Idx_t].x, 2) + pow(cloud_s_trans->points[Idx_s].y - cloud_t->points[Idx_t].y, 2)
			+ pow(cloud_s_trans->points[Idx_s].z - cloud_t->points[Idx_t].z, 2);
		dist = sqrt(dist);
		if (dist <= correct_thresh)
			ids[i] = 1;
		else
			ids[i] = 0;
	}
}
double OTSU_thresh(/*vector<Vote> Vote_score*/Eigen::VectorXd values)
{
	/*vector<double>values;
	for (size_t i = 0; i < Vote_score.size(); i++)
	{
		values.push_back(Vote_score[i].score);
	}*/
	int i;
	int Quant_num = 100;
	double score_sum = 0.0;
	double fore_score_sum = 0.0;
	vector<int> score_Hist(Quant_num, 0);
	vector<double> score_sum_Hist(Quant_num, 0.0);
	double max_score_value, min_score_value;
	vector<double> all_scores;
	for (i = 0; i < values.size(); i++)
	{
		score_sum += values[i];
		all_scores.push_back(values[i]);
	}
	sort(all_scores.begin(), all_scores.end());
	max_score_value = all_scores[all_scores.size() - 1];
	min_score_value = all_scores[0];
	double Quant_step = (max_score_value - min_score_value) / Quant_num;
	for (i = 0; i < values.size(); i++)
	{
		int ID = values[i] / Quant_step;
		if (ID >= Quant_num) ID = Quant_num - 1;
		score_Hist[ID]++;
		score_sum_Hist[ID] += values[i];
	}
	double fmax = -1000;
	int n1 = 0, n2;
	double m1, m2, sb;
	double thresh = (max_score_value - min_score_value) / 2;//default value
	for (i = 0; i < Quant_num; i++)
	{
		double Thresh_temp = i * (max_score_value - min_score_value) / double(Quant_num);
		n1 += score_Hist[i];
		if (n1 == 0) continue;
		n2 = values.size() - n1;
		if (n2 == 0) break;
		fore_score_sum += score_sum_Hist[i];
		m1 = fore_score_sum / n1;
		m2 = (score_sum - fore_score_sum) / n2;
		sb = (double)n1 * (double)n2 * pow(m1 - m2, 2);
		if (sb > fmax)
		{
			fmax = sb;
			thresh = Thresh_temp;
		}
	}
	return thresh;
}

//
double Distance(pcl::PointXYZ& A, pcl::PointXYZ& B) {
	double distance = 0;
	double d_x = (double)A.x - (double)B.x;
	double d_y = (double)A.y - (double)B.y;
	double d_z = (double)A.z - (double)B.z;
	distance = sqrt(d_x * d_x + d_y * d_y + d_z * d_z);
	return distance;
}

double Distance_3DMatch(Vertex A, Vertex B) {
	double distance = 0;
	double d_x = (double)A.x - (double)B.x;
	double d_y = (double)A.y - (double)B.y;
	double d_z = (double)A.z - (double)B.z;
	distance = sqrt(d_x * d_x + d_y * d_y + d_z * d_z);
	return distance;
}

void boost_rand(int seed, int start, int end, int rand_num, std::vector<int>& idx)
{
	boost::mt19937 engine(seed);
	boost::uniform_int<> distribution(start, end);
	boost::variate_generator<boost::mt19937, boost::uniform_int<> > myrandom(engine, distribution);
	std::unordered_set<int> r;
	while (r.size() < rand_num)
	{
		r.insert(myrandom());
	}
	for (auto it = r.begin(); it != r.end(); it++)
	{
		idx.push_back(*it);
	}
}
void Rand_3(int seed, int scale, int& output1, int& output2, int& output3)
{
	std::vector<int> result;
	int start = 0;
	int end = scale - 1;
	boost_rand(seed, start, end, scale, result);
	output1 = result[0];
	output2 = result[1];
	output3 = result[2];
}
void Rand_2(int seed, int scale, int& output1, int& output2)
{
	std::vector<int> result;
	int start = 0;
	int end = scale - 1;
	boost_rand(seed, start, end, scale, result);
	output1 = result[0];
	output2 = result[1];
}
void Rand_1(int seed, int scale, int& output)
{
	std::vector<int> result;
	int start = 0;
	int end = scale - 1;
	boost_rand(seed, start, end, scale, result);
	output = result[0];
}
//Hypothesis quality estimation
void RANSAC_trans_est(pcl::PointXYZ& point_s1, pcl::PointXYZ& point_s2, pcl::PointXYZ& point_s3,
	pcl::PointXYZ& point_t1, pcl::PointXYZ& point_t2, pcl::PointXYZ& point_t3, Eigen::Matrix4f& Mat)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr LRF_source(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr LRF_target(new pcl::PointCloud<pcl::PointXYZ>);

	LRF_source->points.push_back(point_s1); LRF_source->points.push_back(point_s2); LRF_source->points.push_back(point_s3);//LRF_source->points.push_back(s_4);
	LRF_target->points.push_back(point_t1); LRF_target->points.push_back(point_t2); LRF_target->points.push_back(point_t3);//LRF_source->points.push_back(t_4);
	//
	pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> SVD;
	SVD.estimateRigidTransformation(*LRF_source, *LRF_target, Mat);
}

int RANSAC_inliers(PointCloudPtr source_match_points, PointCloudPtr target_match_points, Eigen::Matrix4f& Mat, float correct_thresh)
{
	int i;
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points_trans(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*source_match_points, *source_match_points_trans, Mat);
	//
	int N = 0;
	for (i = 0; i < source_match_points_trans->points.size(); i++)
	{
		float X = source_match_points_trans->points[i].x - target_match_points->points[i].x;
		float Y = source_match_points_trans->points[i].y - target_match_points->points[i].y;
		float Z = source_match_points_trans->points[i].z - target_match_points->points[i].z;
		float dist = sqrt(X * X + Y * Y + Z * Z);
		if (dist < correct_thresh)N++;
	}
	return N;
}

/******************************************************************************************************************************************************/
float Score_est(pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points, pcl::PointCloud<pcl::PointXYZ>::Ptr target_match_points, Eigen::Matrix4f Mat, float thresh, string loss)
{
	int i;
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points_trans(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*source_match_points, *source_match_points_trans, Mat);
	//
	float score = 0;
	/*if (loss.compare("Inlier") == 0)
	{
		score = RANSAC_inliers(source_match_points, target_match_points, Mat, thresh);
	}*/
	/*else
	{*/
	for (i = 0; i < source_match_points_trans->points.size(); i++)
	{
		float X = source_match_points_trans->points[i].x - target_match_points->points[i].x;
		float Y = source_match_points_trans->points[i].y - target_match_points->points[i].y;
		float Z = source_match_points_trans->points[i].z - target_match_points->points[i].z;
		float dist = sqrt(X * X + Y * Y + Z * Z);
		float temp_score;
		if (loss.compare("inlier") == 0) {
			if (dist < thresh)
			{
				score += 1;
			}
		}
		else if (loss.compare("MAE") == 0)
		{
			if (dist < thresh)
			{
				temp_score = (thresh - dist) / thresh;
				score += temp_score;
			}
		}
		else if (loss.compare("MSE") == 0) //recommend
		{
			if (dist < thresh)
			{
				temp_score = (dist - thresh) * (dist - thresh) / (thresh * thresh);
				score += temp_score;
			}
		}
		else if (loss.compare("LOG-COSH") == 0)
		{
			if (dist < thresh)
			{
				temp_score = log(cosh(thresh - dist)) / log(cosh(thresh));
				score += temp_score;
			}
		}
		else if (loss.compare("QUANTILE") == 0)
		{
			if (dist < thresh)
			{
				temp_score = 0.9 * (thresh - dist) / thresh;
			}
			else temp_score = 0.1 * (dist - thresh) / dist;
			score += temp_score;
		}
		else if (loss.compare("-QUANTILE") == 0)
		{
			if (dist < thresh) temp_score = 0.9 * (thresh - dist) / thresh;
			else temp_score = 0.1 * (thresh - dist) / dist;
			score += temp_score;
		}
		else if (loss.compare("EXP") == 0)
		{
			if (dist < thresh)
			{
				temp_score = exp(-(pow(dist, 2) / (2 * pow(thresh, 2))));
				score += temp_score;
			}
		}
	}
	//}
	return score;
}

int RANSAC_score(vector<Corre_3DMatch> Match, float resolution, int  _Iterations, float threshold, Eigen::Matrix4f& Mat, string loss)
{
	Mat = Eigen::Matrix4f::Identity();
	float RANSAC_inlier_judge_thresh = threshold * resolution;
	float score = 0;
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr target_match_points(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < Match.size(); i++)
	{
		pcl::PointXYZ point_s, point_t;
		point_s = Match[i].src;
		point_t = Match[i].des;
		source_match_points->points.push_back(point_s);
		target_match_points->points.push_back(point_t);
	}
	//
	int Iterations = _Iterations;
	int Rand_seed = Iterations;
	int Match_Idx1, Match_Idx2, Match_Idx3;
	while (Iterations)
	{
		Rand_seed--;

		Rand_3(Rand_seed, Match.size(), Match_Idx1, Match_Idx2, Match_Idx3);
		pcl::PointXYZ point_s1, point_s2, point_s3, point_t1, point_t2, point_t3;
		point_s1 = Match[Match_Idx1].src;
		point_s2 = Match[Match_Idx2].src;
		point_s3 = Match[Match_Idx3].src;
		point_t1 = Match[Match_Idx1].des;
		point_t2 = Match[Match_Idx2].des;
		point_t3 = Match[Match_Idx3].des;
		//
		Eigen::Matrix4f Mat_iter;
		RANSAC_trans_est(point_s1, point_s2, point_s3, point_t1, point_t2, point_t3, Mat_iter);
		float score_iter = Score_est(source_match_points, target_match_points, Mat_iter, RANSAC_inlier_judge_thresh, loss);
		if (score_iter > score)
		{
			//cout << "Iterations="<< Iterations<<"inliers_iter=" << inliers_iter << endl;
			//std::cout << "acc=" << (float)((float)inliers_iter / (float)Match.size()) << endl;
			score = score_iter;
			Mat = Mat_iter;
		}
		Iterations--;
	}
	//cout << Match_Idx1 << " " << Match_Idx2 << " " << Match_Idx3 << endl;
	return 1;
}
/*************************************************************************************************************************************************************/

int RANSAC(vector<Corre_3DMatch> Match, float resolution, int  _Iterations, Eigen::Matrix4f& Mat)
{
	Mat = Eigen::Matrix4f::Identity();
	float RANSAC_inlier_judge_thresh = resolution;
	int inliers = 0;
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr target_match_points(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < Match.size(); i++)
	{
		pcl::PointXYZ point_s, point_t;
		point_s = Match[i].src;
		point_t = Match[i].des;
		source_match_points->points.push_back(point_s);
		target_match_points->points.push_back(point_t);
	}
	//
	int Iterations = _Iterations;
	int Rand_seed = Iterations;
	while (Iterations)
	{
		Rand_seed--;
		int Match_Idx1, Match_Idx2, Match_Idx3;
		Rand_3(Rand_seed, Match.size(), Match_Idx1, Match_Idx2, Match_Idx3);
		pcl::PointXYZ point_s1, point_s2, point_s3, point_t1, point_t2, point_t3;
		point_s1 = Match[Match_Idx1].src;
		point_s2 = Match[Match_Idx2].src;
		point_s3 = Match[Match_Idx3].src;
		point_t1 = Match[Match_Idx1].des;
		point_t2 = Match[Match_Idx2].des;
		point_t3 = Match[Match_Idx3].des;
		//
		Eigen::Matrix4f Mat_iter;
		RANSAC_trans_est(point_s1, point_s2, point_s3, point_t1, point_t2, point_t3, Mat_iter);
		int inliers_iter = RANSAC_inliers(source_match_points, target_match_points, Mat_iter, RANSAC_inlier_judge_thresh);
		if (inliers_iter > inliers)
		{
			//cout << "Iterations="<< Iterations<<"inliers_iter=" << inliers_iter << endl;
			//std::cout << "acc=" << (float)((float)inliers_iter / (float)Match.size()) << endl;
			inliers = inliers_iter;
			Mat = Mat_iter;
		}
		Iterations--;
	}
	return 1;
}

double trMatrix(double matrix[][3]) {
	double tr = 0;
	for (int i = 0; i < 3; i++)
	{
		tr += matrix[i][i];
	}
	return tr;
}

double Rotation_error(double gt_R[][4], double est_R[][4])
{
	double res = 0;
	double R[3][3];
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			R[i][j] = (gt_R[i][0] * est_R[0][j] + gt_R[i][1] * est_R[1][j] + gt_R[i][2] * est_R[2][j]);
		}
	}
	//cout << trMatrix(R)<<endl;
	double tr = trMatrix(R);
	if (tr > 3)tr = 3;
	if (tr < -1)tr = -1;
	//return trMatrix(R);
	//return (trMatrix(R) - 1) / 2;
	//return 180/M_PI*acos((trMatrix(R)-1)/2);
	return 180 / M_PI * acos((tr - 1) / 2);

}

double Translation_error(double gt_T[][4], double est_T[][4])
{
	double res = 0;
	for (int i = 0; i < 3; i++)
	{
		res += pow(gt_T[i][3] - est_T[i][3], 2);
	}
	return sqrt(res) * 100;
}