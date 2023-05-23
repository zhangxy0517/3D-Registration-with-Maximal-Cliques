//MKL boost
//#define EIGEN_USE_MKL_ALL
//#define EIGEN_VACTORIZE_SSE4_2
#include <cstdio>
#include <vector>
#include <time.h>
#include <algorithm>
#include <pcl/point_types.h>
#include <pcl/registration/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>
#include <cmath>
#include "Eva.h"
#include "omp.h"
//#include<mkl.h>
#include <unsupported/Eigen/MatrixFunctions>
using namespace Eigen;

bool compare_vote_score(const Vote& v1, const Vote& v2) {
	return v1.score > v2.score;
}

bool compare_vote_degree(const Vote_exp& v1, const Vote_exp& v2) {
	return v1.degree > v2.degree;
}

bool compare_corres_score(const Corre_3DMatch& c1, const Corre_3DMatch& c2) {
	return c1.score > c2.score;
}

bool CV_SortByScore(const Corre& v1, const Corre& v2)
{
	return v1.score > v2.score;
}

double getAngleTwoVectors(const Eigen::Vector3f& v1, const Eigen::Vector3f& v2) {
	double radian_angle = atan2(v1.cross(v2).norm(), v1.transpose() * v2);
	return radian_angle;   //[0,PI]
}

void computeCentroidAndCovariance(Corre_3DMatch& c, PointCloudPtr& src_knn, PointCloudPtr& des_knn) {
	pcl::ConstCloudIterator<pcl::PointXYZ> src_it(*src_knn);
	pcl::ConstCloudIterator<pcl::PointXYZ> des_it(*des_knn);
	src_it.reset(); des_it.reset();
	Eigen::Vector4f centroid_src, centroid_des;
	pcl::compute3DCentroid(src_it, centroid_src);
	pcl::compute3DCentroid(des_it, centroid_des);
	src_it.reset(); des_it.reset();
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> src_demean, des_demean;
	pcl::demeanPointCloud(src_it, centroid_src, src_demean);
	pcl::demeanPointCloud(des_it, centroid_des, des_demean);
	Eigen::Matrix3f src_cov = (src_demean * src_demean.transpose());
	Eigen::Matrix3f des_cov = (des_demean * des_demean.transpose());
	c.centeroid_src = centroid_src;
	c.centeroid_des = centroid_des;
	c.covariance_src = src_cov;
	c.covariance_des = des_cov;
}

float wasserstein_dis(Corre_3DMatch& c1, Corre_3DMatch &c2) 
{
	float src_wasserstein = (c1.covariance_src + c2.covariance_src - 2 * ((c1.covariance_src).sqrt() * c2.covariance_src * (c1.covariance_src).sqrt()).sqrt()).trace();
	float des_wasserstein = (c1.covariance_des + c2.covariance_des - 2 * ((c1.covariance_des).sqrt() * c2.covariance_des * (c1.covariance_des).sqrt()).sqrt()).trace();
	src_wasserstein += (c1.centeroid_src - c2.centeroid_src).norm();
	des_wasserstein += (c1.centeroid_des - c2.centeroid_des).norm();
	float dis = exp(-pow(1 - src_wasserstein / des_wasserstein, 2));
	if (dis < 0 || dis > 1) dis = 0;
	return dis;
}

Eigen::MatrixXf Graph_construction(vector<Corre_3DMatch>& correspondence, float resolution, bool sc2, const string &name, const string &descriptor) {
	int size = correspondence.size();
	Eigen::MatrixXf cmp_score;
	cmp_score.resize(size, size);
	cmp_score.setZero();
	Corre_3DMatch c1, c2;
	float score, src_dis, des_dis, dis, alpha_dis;
	if (name == "KITTI")
	{
        float thresh = descriptor == "fpfh" ? 0.9 : 0.999;
		for (int i = 0; i < size; i++)
		{
			c1 = correspondence[i];
			for (int j = i + 1; j < size; j++)
			{
				c2 = correspondence[j];
				src_dis = Distance(c1.src, c2.src);
				des_dis = Distance(c1.des, c2.des);
				dis = abs(src_dis - des_dis);
				score = 1 - (dis * dis) / (0.6 * 0.6);
				//score = exp(-dis * dis);
				score = (score < thresh) ? 0 : score;//fcgf 0.9999 fpfh 0.9
				cmp_score(i, j) = score;
				cmp_score(j, i) = score;

			}
		}
	}
	else if (name == "3dmatch" || name == "3dlomatch")
	{
		for (int i = 0; i < size; i++)
		{
			c1 = correspondence[i];
			for (int j = i + 1; j < size; j++)
			{
				c2 = correspondence[j];
				src_dis = Distance(c1.src, c2.src);
				des_dis = Distance(c1.des, c2.des);
				dis = abs(src_dis - des_dis);

				if (descriptor == "predator" || low_inlieratio)
				{
					score = 1 - (dis * dis) / (0.1 * 0.1);
					if (add_overlap || low_inlieratio)
					{
                        score = (score < 0.99) ? 0 : score; //fpfh/fcgf overlap 0.99
//                        else {
//                            alpha_dis = 10 * resolution;
//                            score = exp(-dis * dis / (2 * alpha_dis * alpha_dis));
//                            score = (score < 0.999) ? 0 : score;
//                        }
// mac-op 250 500 1000 2500 5000
//        0.9 0.95 0.99 0.995 0.999
					}
					else {
						score = (score < 0.999) ? 0 : score;
					}
				}
				else {
					alpha_dis = 10 * resolution;
					score = exp(-dis * dis / (2 * alpha_dis * alpha_dis));
					if (name == "3dmatch" && descriptor == "fcgf")
					{
						score = (score < 0.999) ? 0 : score;
					}
					else if (name == "3dmatch" && descriptor == "fpfh") {
						score = (score < 0.995) ? 0 : score;
					}
					else if (descriptor == "spinnet" || descriptor == "d3feat") {
						score = (score < 0.85) ? 0 : score;
                        // spinnet 5000 2500 1000 500 250
                        //         0.99 0.99 0.95 0.9 0.85
					}
					else {
						score = (score < 0.99) ? 0 : score; //3dlomatch 0.99, 3dmatch fcgf 0.999 fpfh 0.995
					}
				}

				cmp_score(i, j) = score;
				cmp_score(j, i) = score;
			}
		}
	}
	else if (name == "U3M") {
		for (int i = 0; i < size; i++)
		{
			c1 = correspondence[i];
			for (int j = i + 1; j < size; j++)
			{
				c2 = correspondence[j];
				src_dis = Distance(c1.src, c2.src);
				des_dis = Distance(c1.des, c2.des);
				dis = abs(src_dis - des_dis);
				alpha_dis = 10 * resolution;
				score = exp(-dis * dis / (2 * alpha_dis * alpha_dis));
				score = (score < 0.99) ? 0 : score;
				cmp_score(i, j) = score;
				cmp_score(j, i) = score;
			}
		}
	}
	if (sc2)
	{
		//Eigen::setNbThreads(6);
		cmp_score = cmp_score.cwiseProduct(cmp_score * cmp_score);
	}
	return cmp_score;
	}

Eigen::MatrixXf Graph_construction(vector<Corre_3DMatch>& correspondence, float resolution, bool sc2, float cmp_thresh) {
    int size = correspondence.size();
    Eigen::MatrixXf cmp_score; 
    cmp_score.resize(size, size);
    cmp_score.setZero();
    Corre_3DMatch c1, c2;
    float score, src_dis, des_dis, dis, alpha_dis;

    for (int i = 0; i < size; i++)
    {
        c1 = correspondence[i];
        for (int j = i + 1; j < size; j++)
        {
            c2 = correspondence[j];
            src_dis = Distance(c1.src, c2.src);
            des_dis = Distance(c1.des, c2.des);
            dis = abs(src_dis - des_dis);
            alpha_dis = 10 * resolution;
            score = exp(-dis * dis / (2 * alpha_dis * alpha_dis));
            score = (score < cmp_thresh) ? 0 : score;
            cmp_score(i, j) = score;
            cmp_score(j, i) = score;
}
    }
    if (sc2)
    {
        //Eigen::setNbThreads(6);
        cmp_score = cmp_score.cwiseProduct(cmp_score * cmp_score);
    }
    return cmp_score;
}

vector<int> vectors_intersection(vector<int> v1, vector<int> v2) {
	vector<int> v;
	set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), back_inserter(v));
	return v;
}

double calculate_rotation_error(Eigen::Matrix3d& est, Eigen::Matrix3d& gt) {
	double tr = (est.transpose() * gt).trace();
	return acos(min(max((tr - 1.0) / 2.0, -1.0), 1.0)) * 180.0 / M_PI;
}

double calculate_translation_error(Eigen::Vector3d& est, Eigen::Vector3d& gt) {
	Eigen::Vector3d t = est - gt;
	return sqrt(t.dot(t)) * 100;
}

bool evaluation_est(Eigen::Matrix4d est, Eigen::Matrix4d gt, double re_thresh, double te_thresh, double& RE, double& TE) {
	Eigen::Matrix3d rotation_est, rotation_gt;
	Eigen::Vector3d translation_est, translation_gt;
	rotation_est = est.topLeftCorner(3, 3);
	rotation_gt = gt.topLeftCorner(3, 3);
	translation_est = est.block(0, 3, 3, 1);
	translation_gt = gt.block(0, 3, 3, 1);

	RE = calculate_rotation_error(rotation_est, rotation_gt);
	TE = calculate_translation_error(translation_est, translation_gt);
	if (0 <= RE && RE <= re_thresh && 0 <= TE && TE <= te_thresh)
	{
		return true;
	}
	return false;
}

void sort_row(MatD& matrix, MatD& sorted_matrix, Eigen::MatrixXi& index) {
	sorted_matrix.resize(matrix.rows(), matrix.cols());
	index.resize(matrix.rows(), matrix.cols());
	for (int n = 0; n < matrix.rows(); n++)
	{
		VectorXi row_index = VectorXi::LinSpaced(matrix.cols(), 0, matrix.cols() - 1);
		VectorXd row_data = matrix.row(n);
		auto rule = [row_data](int i, int j)->bool {
			return row_data(i) > row_data(j);
		};
		sort(row_index.data(), row_index.data() + row_index.size(), rule);

		for (int i = 0; i < row_data.size(); i++)
		{
			sorted_matrix(n, i) = row_data(row_index(i));
		}
		index.row(n) = row_index;
	}
}

void weight_SVD(PointCloudPtr& src_pts, PointCloudPtr& des_pts, Eigen::VectorXd& weights, double weight_threshold, Eigen::Matrix4d& trans_Mat) {
	for (size_t i = 0; i < weights.size(); i++)
	{
		weights(i) = (weights(i) < weight_threshold) ? 0 : weights(i);
	}
	
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> weight;
	Eigen::VectorXd ones = weights;
	ones.setOnes();
	weight = (weights * ones.transpose());
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Identity = weight;
	
	Identity.setIdentity();
	weight = (weights * ones.transpose()).cwiseProduct(Identity);
	pcl::ConstCloudIterator<pcl::PointXYZ> src_it(*src_pts);
	pcl::ConstCloudIterator<pcl::PointXYZ> des_it(*des_pts);
	
	src_it.reset(); des_it.reset();
	Eigen::Matrix<double, 4, 1> centroid_src, centroid_des;
	pcl::compute3DCentroid(src_it, centroid_src);
	pcl::compute3DCentroid(des_it, centroid_des);

	src_it.reset(); des_it.reset();
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> src_demean, des_demean;
	pcl::demeanPointCloud(src_it, centroid_src, src_demean);
	pcl::demeanPointCloud(des_it, centroid_des, des_demean);

	Eigen::Matrix<double, 3, 3> H = (src_demean * weight * des_demean.transpose()).topLeftCorner(3, 3);

	// Compute the Singular Value Decomposition
	Eigen::JacobiSVD<Eigen::Matrix<double, 3, 3> > svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix<double, 3, 3> u = svd.matrixU();
	Eigen::Matrix<double, 3, 3> v = svd.matrixV();

	// Compute R = V * U'
	if (u.determinant() * v.determinant() < 0)
	{
		for (int x = 0; x < 3; ++x)
			v(x, 2) *= -1;
	}

	Eigen::Matrix<double, 3, 3> R = v * u.transpose();

	// Return the correct transformation
	Eigen::Matrix<double, 4, 4> Trans;
	Trans.setIdentity();
	Trans.topLeftCorner(3, 3) = R;
	const Eigen::Matrix<double, 3, 1> Rc(R * centroid_src.head(3));
	Trans.block(0, 3, 3, 1) = centroid_des.head(3) - Rc;
	trans_Mat = Trans;
}

void post_refinement(vector<Corre_3DMatch>&correspondence, PointCloudPtr& src_corr_pts, PointCloudPtr& des_corr_pts, Eigen::Matrix4d& initial, double& best_score, double inlier_thresh, int iterations, const string &metric) {
	int pointNum = src_corr_pts->points.size();
	double pre_score = best_score;
	for (int i = 0; i < iterations; i++)
	{
		double score = 0;
		Eigen::VectorXd weights, weight_pred;
		weights.resize(pointNum);
		weights.setZero();
		vector<int> pred_inlier_index;
		PointCloudPtr trans(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::transformPointCloud(*src_corr_pts, *trans, initial);
		for (int j = 0; j < pointNum; j++)
		{
			double dist = Distance(trans->points[j], des_corr_pts->points[j]);
			double w = 1;
			if (add_overlap)
			{
				w = correspondence[j].score;
			}
			if (dist < inlier_thresh)
			{
				pred_inlier_index.push_back(j);
				weights[j] = 1 / (1 + pow(dist / inlier_thresh, 2));
				if (metric == "inlier")
				{
					score+=1*w;
				}
				else if (metric == "MAE")
				{
					score += (inlier_thresh - dist)*w / inlier_thresh;
				}
				else if (metric == "MSE")
				{
					score += pow((inlier_thresh - dist), 2)*w / pow(inlier_thresh, 2);
				}
			}
		}
		if (score < pre_score) {
			break;
		}
		else {
			pre_score = score;
			//估计pred_inlier
			PointCloudPtr pred_src_pts(new pcl::PointCloud<pcl::PointXYZ>);
			PointCloudPtr pred_des_pts(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::copyPointCloud(*src_corr_pts, pred_inlier_index, *pred_src_pts);
			pcl::copyPointCloud(*des_corr_pts, pred_inlier_index, *pred_des_pts);
			weight_pred.resize(pred_inlier_index.size());
			for (int k = 0; k < pred_inlier_index.size(); k++)
			{
				weight_pred[k] = weights[pred_inlier_index[k]];
			}
			//weighted_svd
			weight_SVD(pred_src_pts, pred_des_pts, weight_pred, 0, initial);
			pred_src_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
			pred_des_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
		}
		pred_inlier_index.clear();
		trans.reset(new pcl::PointCloud<pcl::PointXYZ>);
	}
	best_score = pre_score;
}

double evaluation_trans(vector<Corre_3DMatch>& Match, vector<Corre_3DMatch>& correspondnece, PointCloudPtr& src_corr_pts, PointCloudPtr& des_corr_pts, double weight_thresh, Eigen::Matrix4d& trans, double metric_thresh, const string &metric, float resolution, bool instance_equal) {

	PointCloudPtr src_pts(new pcl::PointCloud<pcl::PointXYZ>);
	PointCloudPtr des_pts(new pcl::PointCloud<pcl::PointXYZ>);
	vector<double>weights;
	for (auto & i : Match)
	{
		if (i.score >= weight_thresh)
		{
			src_pts->push_back(i.src);
			des_pts->push_back(i.des);
			weights.push_back(i.score);
		}
	}
	if (weights.size() < 3)
	{
		return 0;
	}
	Eigen::VectorXd weight_vec = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(weights.data(), weights.size());
	weights.clear();
    weights.shrink_to_fit();
	weight_vec /= weight_vec.maxCoeff();
	if (!add_overlap || instance_equal) {
		weight_vec.setOnes(); // 2023.2.23 
	}
	weight_SVD(src_pts, des_pts, weight_vec, 0, trans);
	PointCloudPtr src_trans(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*src_corr_pts, *src_trans, trans);
	//Eigen::Matrix4f trans_f = trans.cast<float>();
	//Eigen::Matrix3f R = trans_f.topLeftCorner(3, 3);
	double score = 0.0;
	int inlier = 0;
		int corr_num = src_corr_pts->points.size();
	for (int i = 0; i < corr_num; i++)
		{
			double dist = Distance(src_trans->points[i], des_corr_pts->points[i]);
		double w = 1;
		if (add_overlap)
		{
			w = correspondnece[i].score;
		}
		if (dist < metric_thresh)
		{
			inlier++;
			if (metric == "inlier")
			{
				score += 1*w;//correspondence[i].inlier_weight;
			}
			else if (metric == "MAE")
			{
				score += (metric_thresh - dist)*w / metric_thresh;
			}
			else if (metric == "MSE")
			{
				score += pow((metric_thresh - dist), 2)*w / pow(metric_thresh, 2);
			}
		}
	}
	src_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
	des_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
	src_trans.reset(new pcl::PointCloud<pcl::PointXYZ>);
	return score;
}


void eigenvector_centrality(Eigen::MatrixXd& Graph, Eigen::VectorXd& initial, Eigen::VectorXd& eigenvector) {
	eigenvector.resize(initial.size());
	eigenvector = initial;
	Eigen::VectorXd eigenvector_next = eigenvector;
	eigenvector_next.setZero();
	double max = 0;
	bool flag = false;

	Eigen::MatrixXd zero_one = Graph;
	for (int i = 0; i < Graph.rows(); i++)
	{
		for (int j = 0; j < Graph.cols(); j++) {
			zero_one(i, j) = Graph(i, j) ? 1 : 0;
		}
	}

	double tmp_max;
	while (!flag) {
		tmp_max = max;
		eigenvector_next = zero_one * eigenvector;
		//cout << eigenvector_next << endl;
		max = eigenvector_next.maxCoeff();
		cout << max << endl;
		for (int i = 0; i < eigenvector.size(); i++)
		{
			if (eigenvector(i) != eigenvector_next(i)) {
				break;
			}
			if (i == eigenvector.size() - 1)
			{
				flag = true;
			}
		}
		if (abs(1.0 / tmp_max - 1.0 / max) < 0.01)
		{
			flag = true;
		}
		eigenvector = eigenvector_next;
		eigenvector_next.setZero();
	}
	eigenvector /= max;
}

int print_vector(const igraph_vector_t* v) {
	long int i, n = igraph_vector_size(v);
	for (i = 0; i < n; i++) {
		printf("%ld %.2g\n", i, (double)VECTOR(*v)[i]);
	}
	return 0;
}

void print_graph(igraph_t* g) {
	igraph_vector_t el;
	long int i, j, n;
	char ch = igraph_is_directed(g) ? '>' : '-';

	igraph_vector_init(&el, 0);
	igraph_get_edgelist(g, &el, 0);
	n = igraph_ecount(g);

	for (i = 0, j = 0; i < n; i++, j += 2) {
		printf("%ld --%c %ld: %ld\n",
			(long)VECTOR(el)[j], ch, (long)VECTOR(el)[j + 1], (long)EAN(g, "weight", i));
	}
	printf("\n");

	igraph_vector_destroy(&el);
}

int sort_cmp(const void* a, const void* b) {
	const igraph_vector_t** da = (const igraph_vector_t**)a;
	const igraph_vector_t** db = (const igraph_vector_t**)b;
	int i, alen = igraph_vector_size(*da), blen = igraph_vector_size(*db);
	if (alen != blen) {
		return (alen < blen) - (alen > blen);
	}
	for (i = 0; i < alen; i++) {
		int ea = VECTOR(**da)[i], eb = VECTOR(**db)[i];
		if (ea != eb) {
			return (ea > eb) - (ea < eb);
		}
	}
	return 0;
}

void sort_cliques(igraph_vector_ptr_t* cliques) {
	int i, n = igraph_vector_ptr_size(cliques);
	for (i = 0; i < n; i++) {
		igraph_vector_t* v = (igraph_vector_t*)VECTOR(*cliques)[i];
		igraph_vector_sort(v);
	}
	igraph_qsort(VECTOR(*cliques), (size_t)n,
		sizeof(igraph_vector_t*), sort_cmp);
}

void print_and_destroy_cliques(igraph_vector_ptr_t* cliques) {
	int i;
	sort_cliques(cliques);
	for (i = 0; i < igraph_vector_ptr_size(cliques); i++) {
		igraph_vector_t* v = (igraph_vector_t*)VECTOR(*cliques)[i];
		igraph_vector_print(v);
		igraph_vector_destroy(v);
		igraph_free(v);
	}
}

void find_largest_clique_of_node(Eigen::MatrixXf& Graph, igraph_vector_ptr_t* cliques, vector<Corre_3DMatch>& correspondence, node_cliques* result, vector<int>& remain, int num_node, int est_num, string descriptor) {
	int* vis = new int[igraph_vector_ptr_size(cliques)];
	memset(vis, 0, igraph_vector_ptr_size(cliques));
#pragma omp parallel for
	for (int i = 0; i < num_node; i++)
	{
		result[i].clique_index = -1;
		result[i].clique_size = 0;
		result[i].clique_weight = 0;
		result[i].clique_num = 0;
	}

	for (int i = 0; i < remain.size(); i++)
	{
		igraph_vector_t* v = (igraph_vector_t*)VECTOR(*cliques)[remain[i]];
		float weight = 0;
		int length = igraph_vector_size(v);
		for (int j = 0; j < length; j++)
		{
			int a = (int)VECTOR(*v)[j];
			for (int k = j + 1; k < length; k++)
			{
				int b = (int)VECTOR(*v)[k];
				weight += Graph(a, b);
			}
		}
		for (int j = 0; j < length; j++)
		{
			int k = (int)VECTOR(*v)[j];
			if (result[k].clique_weight < weight)
			{
				result[k].clique_index = remain[i];
				vis[remain[i]]++;
				result[k].clique_size = length;
				result[k].clique_weight = weight;
			}
		}
	}

#pragma omp parallel for
	for (int i = 0; i < remain.size(); i++)
	{
		if (vis[remain[i]] == 0) {
			igraph_vector_t* v = (igraph_vector_t*)VECTOR(*cliques)[remain[i]];
			igraph_vector_destroy(v);
		}
	}

	vector<int>after_delete;
	for (int i = 0; i < num_node; i++)
	{
		if (result[i].clique_index < 0)
		{
			continue;
		}
		if (vis[result[i].clique_index] > 0)
		{
			vis[result[i].clique_index] = 0;
			after_delete.push_back(result[i].clique_index);
		}
		else if (vis[result[i].clique_index] == 0) {
			result[i].clique_index = -1;
		}
	}
	remain.clear();
	remain = after_delete;

	// Normal consistency
//	vector<int>after_selection;
//#pragma omp parallel for
//	for (int i = 0; i < num_node; i++)
//	{
//		if (result[i].clique_index < 0)
//		{
//			continue;
//		}
//		igraph_vector_t* v = (igraph_vector_t*)VECTOR(*cliques)[result[i].clique_index];
//		int length = igraph_vector_size(v);
//		Eigen::VectorXi angle_cmp_vector;
//		angle_cmp_vector.resize(length);
//		angle_cmp_vector.setZero();
//		for (int j = 0; j < length; j++)
//		{
//			int a = (int)VECTOR(*v)[j];
//			for (int k = j + 1; k < length; k++) {
//				int b = (int)VECTOR(*v)[k];
//				float angle_src = getAngleTwoVectors(correspondence[a].src_norm, correspondence[b].src_norm);
//				float angle_des = getAngleTwoVectors(correspondence[a].des_norm, correspondence[b].des_norm);
//				float angle_cmp = abs(sin(angle_src) - sin(angle_des));
//				if (angle_cmp < 0.1)
//				{
//					angle_cmp_vector[k]++;
//					angle_cmp_vector[j]++;
//				}
//			}
//		}
//		int sum = angle_cmp_vector.sum();
//#pragma omp critical
//		{
//			if (sum > length * (length - 1) / 2) //fpfh 0.1 fcgf0.05
//			{
//				after_selection.push_back(result[i].clique_index);
//			}
//			else
//			{
//				result[i].clique_index = -1;
//				//igraph_vector_destroy(v);
//			}
//		}
//	}
//	remain.clear();
//	remain = after_selection;
//	after_selection.clear();

	//reduce the number of cliques
	if (remain.size() > est_num)
	{
		vector<int>after_decline;
		vector<Vote>clique_score;
		for (int i = 0; i < num_node; i++)
		{
			if (result[i].clique_index < 0)
			{
				continue;
			}
			Vote t;
			t.index = result[i].clique_index;
			t.score = result[i].clique_weight;
			clique_score.push_back(t);
		}
		sort(clique_score.begin(), clique_score.end(), compare_vote_score);
		for (int i = 0; i < est_num; i++)
		{
			after_decline.push_back(clique_score[i].index);
		}
		remain.clear();
		remain = after_decline;
        clique_score.clear();
	}
    delete[] vis;
	return;
}

int Iter_trans_est(PointCloudPtr& cloud_source, PointCloudPtr& cloud_target, float& mr, float& inlier_thresh, vector<int>& Sample_cloud_Idx, float& residual_error, Eigen::Matrix4f& Mat)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr closet_source(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr closet_target(new pcl::PointCloud<pcl::PointXYZ>);
	//
	residual_error = 0;
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	vector<int> Idx;
	vector<float> Dist;
	kdtree.setInputCloud(cloud_target);
	for (int i = 0; i < Sample_cloud_Idx.size(); i++)
	{
		kdtree.nearestKSearch(cloud_source->points[Sample_cloud_Idx[i]], 1, Idx, Dist);
		if (sqrt(Dist[0]) <= inlier_thresh * mr)
		{
			closet_source->points.push_back(cloud_source->points[Sample_cloud_Idx[i]]);
			closet_target->points.push_back(cloud_target->points[Idx[0]]);
			residual_error += sqrt(Dist[0]);
		}
	}
	if (closet_source->points.size() == 0)
		return -1;
	else
	{
		residual_error /= closet_source->points.size();
		residual_error /= mr;
		pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> SVD;
		SVD.estimateRigidTransformation(*closet_source, *closet_target, Mat);
	}
	return 0;
}

void GUO_ICP(PointCloudPtr& cloud_source, PointCloudPtr& cloud_target, float& mr, int& Max_iter_Num, Eigen::Matrix4f& Mat_ICP)
{
	int number_of_sample_points;
	float residual_error = 4.0f;
	float inlier_thresh = 4.0f;
	Mat_ICP = Eigen::Matrix4f::Identity();
	for (int i = 0; i < Max_iter_Num; i++)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source_trans(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::transformPointCloud(*cloud_source, *cloud_source_trans, Mat_ICP);
		number_of_sample_points = cloud_source_trans->points.size() / pow(3.0f, residual_error);
		vector<int> Sample_cloud_Idx;
		boost::uniform_int<> distribution(0, cloud_source_trans->points.size());
		boost::mt19937 engine;
		boost::variate_generator<boost::mt19937, boost::uniform_int<> > myrandom(engine, distribution);
		for (int j = 0; j < number_of_sample_points; j++)
			Sample_cloud_Idx.push_back(myrandom());
		Eigen::Matrix4f Mat_i;
		int flag = Iter_trans_est(cloud_source_trans, cloud_target, mr, inlier_thresh, Sample_cloud_Idx, residual_error, Mat_i);
		if (flag == -1)
		{
			printf("阈值过小，无法找到匹配点！\n");
			break;
		}
		Mat_ICP = Mat_i * Mat_ICP;
		if (residual_error <= 0.01) break;
	}
}

void sort_eigenvector(Eigen::VectorXd& eigenvector, Eigen::VectorXd& sorted_eigenvector, Eigen::VectorXi& index_eigenvector) {
	sorted_eigenvector = eigenvector;
	index_eigenvector = VectorXi::LinSpaced(eigenvector.size(), 0, eigenvector.size() - 1);
	auto rule = [eigenvector](int i, int j)->bool {
		return eigenvector(i) > eigenvector(j);
	};
	sort(index_eigenvector.data(), index_eigenvector.data() + index_eigenvector.size(), rule);
	for (size_t i = 0; i < eigenvector.size(); i++)
	{
		sorted_eigenvector(i) = eigenvector(index_eigenvector(i));
	}
}

int GTM_corre_select(int Iterations, float mr, PointCloudPtr& cloud_source, PointCloudPtr& cloud_target,
	vector<Corre_3DMatch> Match, vector<int>& Match_inlier)
{
	int i, j;
	Eigen::MatrixXd M(Match.size(), Match.size());//game matrix
	Eigen::VectorXd V(Match.size());//population
	for (i = 0; i < Match.size(); i++)
	{
		for (j = 0; j < Match.size(); j++)
		{
			if (i == j)
				M(i, j) = 0;
			if (i < j)
			{
				double x1 = Match[i].src.x - Match[j].src.x;
				double y1 = Match[i].src.y - Match[j].src.y;
				double z1 = Match[i].src.z - Match[j].src.z;
				double x2 = Match[i].des.x - Match[j].des.x;
				double y2 = Match[i].des.y - Match[j].des.y;
				double z2 = Match[i].des.z - Match[j].des.z;
				double a = sqrt(pow(x1, 2) + pow(y1, 2) + pow(z1, 2));
				double b = sqrt(pow(x2, 2) + pow(y2, 2) + pow(z2, 2));
				if ((a != 0.0) && (b != 0.0))
				{
					M(i, j) = a / b;
					if (M(i, j) > b / a) M(i, j) = b / a;
				}
				else
					M(i, j) = 0;
			}
			if (i > j)
				M(i, j) = M(j, i);
		}
	}
	//
	for (i = 0; i < Match.size(); i++) V(i) = 1.0 / Match.size();
	for (i = 0; i < Iterations; i++)
	{
		Eigen::VectorXd UP(Match.size());
		UP = M * V;
		double down = V.transpose() * UP;
		for (j = 0; j < Match.size(); j++)
		{
			double up = UP(j);
			V(j) = V(j) * up / down;
		}
	}

	for (i = 0; i < Match.size(); i++)	Match[i].score = V(i);
	/*if (k <= Match.size())
	{
		for (i = 0; i < k; i++)
		{
			Match_inlier.push_back(Match[i]);
		}
	}
	else Match_inlier = Match;
	if (Match.size() == 0) return -1;*/
	/*for (i = 0; i < Match.size(); i++)
	{
		Match_inlier.push_back(Match[i]);
	}
	return 0;*/
	Eigen::VectorXd values;
	values.resize(Match.size());
	for (i = 0; i < Match.size(); i++) {
		values[i] = V(i);
	}
	sort(values.data(), values.data() + values.size());
	double thresh = OTSU_thresh(values);
	for (i = 0; i < Match.size(); i++)
	{
		if (V[i] > thresh) {
			Match_inlier.push_back(1);
		}
		else {
			Match_inlier.push_back(0);
		}
	}
    return 0;
}

int Geometric_consistency(vector<Vote_exp>pts_degree, vector<int>& Match_inlier) {
	for (int i = 0; i < pts_degree.size(); i++)
	{
		Match_inlier.push_back(0);
	}
	int max_degree = pts_degree[0].degree;
	Match_inlier[pts_degree[0].index] = 1;
	for (int i = 0; i < pts_degree[0].degree; i++)
	{
		Match_inlier[pts_degree[0].corre_index[i]] = 1;
	}
	for (int i = 1; i < pts_degree.size(); i++)
	{
		if (pts_degree[i].degree < max_degree)
		{
			break;
		}
		else {
			Match_inlier[pts_degree[i].index] = 1;
			for (int j = 0; j < pts_degree[j].degree; j++)
			{
				Match_inlier[pts_degree[j].corre_index[j]] = 1;
			}
		}
	}
	return 0;
}

bool allclose(Eigen::VectorXf& input, Eigen::VectorXf& other) {
	float atol = 1e-08;
	float rtol = 1e-05;
	if ((input - other).norm() <= atol + rtol * other.norm())
	{
		return true;
	}
	return false;
}

Eigen::VectorXf power_iteration(Eigen::MatrixXf& Graph, int iteration) {
	Eigen::VectorXf leading_eig(Graph.rows());
	leading_eig.setOnes();
	Eigen::VectorXf leading_eig_last = leading_eig;
	for (int i = 0; i < iteration; i++)
	{
		leading_eig = Graph * leading_eig;
		leading_eig = leading_eig / (leading_eig.norm() + 1e-6);
		if (allclose(leading_eig, leading_eig_last)) {
			break;
		}
		leading_eig_last = leading_eig;
	}
	return leading_eig;
}

//保存数据,需要与寻找法向量部分组合
void savetxt(vector<Corre_3DMatch>corr, const string& save_path) {
	ofstream outFile;
	outFile.open(save_path.c_str(), ios::out);
	for (auto & i : corr)
	{
		outFile << i.src_index << " " << i.des_index << endl;
	}
	outFile.close();
}
