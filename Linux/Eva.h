#ifndef _EVA_H_ 
#define _EVA_H_
#define Pi 3.1415926
#define constE 2.718282
#define NULL_POINTID -1
#define NULL_Saliency -1000
#define Random(x) (rand()%x)
#define Corres_view_gap -200
#define Align_precision_threshold 0.1
#define tR 116//30
#define tG 205//144
#define tB 211//255
#define sR 253//209//220
#define sG 224//26//20
#define sB 2//32//60
#define L2_thresh 0.5
#define Ratio_thresh 0.2
#define GC_dist_thresh 3
#define Hough_bin_num 15
#define SI_GC_thresh 0.8
#define RANSAC_Iter_Num 5000
#define GTM_Iter_Num 100
#define CV_voting_size 20
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
using namespace std;
extern bool add_overlap;
extern bool low_inlieratio;
extern bool no_logs;
//
#include <pcl/surface/gp3.h>
#include <pcl/surface/mls.h>
#include <unordered_set>
#include <Eigen/Eigen>
#include <igraph/igraph.h>
#include <sys/stat.h>
#include <unistd.h>
//
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudPtr;
typedef pcl::PointXYZ PointInT;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatD;
typedef struct {
	float x;
	float y;
	float z;
}Vertex;
typedef struct {
	float x;
	float y;
	float z;
	float dist;
	float angle_to_axis;
}Vertex_d_ang;
typedef struct {
	int pointID;
	Vertex x_axis;
	Vertex y_axis;
	Vertex z_axis;
}LRF;
typedef struct {
	int source_idx;
	int target_idx;
	LRF source_LRF;
	LRF target_LRF;
	double score;
}Corre;
typedef struct {
	int src_index;
	int des_index;
	pcl::PointXYZ src;
	pcl::PointXYZ des;
	Eigen::Vector3f src_norm;
	Eigen::Vector3f des_norm;
	Eigen::Matrix3f covariance_src, covariance_des;					//创建3×3协方差矩阵存储对象
	Eigen::Vector4f centeroid_src, centeroid_des;					//创建用于计算协方差矩阵的点云质心对象
	double score;
	int inlier_weight;
}Corre_3DMatch;
typedef struct {
	int PointID;
	float eig1_2;
	float eig2_3;
	float saliency;
	bool TorF;//True or False
}ISS_Key_Type;
typedef struct {
	float M[4][4];
}TransMat;
typedef struct
{
	int index;
	double score;
}Vote;
typedef struct
{
	int index;
	int degree;
	double score;
	vector<int> corre_index;
	int true_num;
}Vote_exp;
typedef struct
{
	vector<int> v;
	int pt1;
	int pt2;
}Intersection_set;
typedef struct
{
	int clique_index;
	int clique_size;
	float clique_weight;
	int clique_num;
}node_cliques;
/**********************************************funcs***************************************/
//dataload
int XYZorMeshlabPly_Read(string Filename, PointCloudPtr& cloud);
int XYZorPly_Read(string Filename, PointCloudPtr& cloud);
void write_cloud(PointCloudPtr cloud, string file_name);
float MeshResolution_mr_compute(PointCloudPtr& cloud);
void Cloud2Meshlab_showfile(string file_open, string file_save);
//int GTMatRead(string &Filename, Eigen::Matrix4d& Mat_GT);
void affinity_matrix_compute(PointCloudPtr cloud_source, PointCloudPtr cloud_target, float mr, vector<Corre> Corres, Eigen::MatrixXf& M);
void find_inlier_corre_id(PointCloudPtr cloud_s, PointCloudPtr cloud_t, vector<Corre> Corres, float correct_thresh, Eigen::Matrix4f& GT_mat, vector<int>& ids);
void feature_matching(PointCloudPtr& cloud_source, PointCloudPtr& cloud_target,
                      vector<vector<float>>& feature_source, vector<vector<float>>& feature_target, vector<Corre_3DMatch>& Corres);
void feature_matching(PointCloudPtr& cloud_source, PointCloudPtr& cloud_target, vector<LRF>LRFs_source, vector<LRF>LRFs_target,
	vector<int>& Idx_source, vector<int>& Idx_target, vector<vector<float>>& feature_source, vector<vector<float>>& feature_target, vector<Corre>& Corres);
void feature_matching_ratio(PointCloudPtr& cloud_source, PointCloudPtr& cloud_target, vector<LRF>LRFs_source, vector<LRF>LRFs_target,
	vector<int>& Idx_source, vector<int>& Idx_target, vector<vector<float>>& feature_source, vector<vector<float>>& feature_target, vector<Corre>& Corres);
void MyType2Eigen(TransMat& M, Eigen::Matrix4f& EigenM);
void Add_Gaussian_noise(float dev, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_noise);
void cloud_simp(float size, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_filtered);
int Correct_corre_compute(PointCloudPtr cloud_s, PointCloudPtr cloud_t, vector<Corre> Corres, float correct_thresh, Eigen::Matrix4d& GT_mat, string path);
void Correct_corre_select(PointCloudPtr cloud_s, PointCloudPtr cloud_t, vector<Corre> Corres, float correct_thresh,
	Eigen::Matrix4f& GT_mat, vector<Corre>& Corres_selected);
double OTSU_thresh(/*vector<Vote> Vote_score*/Eigen::VectorXd values);
double Distance(pcl::PointXYZ& A, pcl::PointXYZ& B);
double Distance_3DMatch(Vertex A, Vertex B);
Eigen::MatrixXf Graph_construction(vector<Corre_3DMatch>& correspondence, float resolution, bool sc2, const string &name,const string &descriptor);
Eigen::MatrixXf Graph_construction(vector<Corre_3DMatch>& correspondence, float resolution, bool sc2, float cmp_thresh);
/**********************************************3DCorres_methods***************************************/
//descriptor
void SHOT_compute(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, vector<int> indices, float sup_radius, vector<vector<float>>& features, vector<LRF>& LRFs);
pcl::PointCloud<pcl::PointXYZ>::Ptr getISS3dKeypoint(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, float resolution, vector<int>& key_indices);
void LFSH_compute(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, vector<int> indices, float sup_radius, int bin_num, vector<vector<float>>& Histograms);
void RCS_compute(pcl::PointCloud<pcl::PointXYZ>::Ptr  cloud, vector<int> indices, float sup_radius, float rotate_angle, int num_of_rotations,
	int num_of_contour_points, vector<vector<float>>& Histograms);
void ISS_detector(PointCloudPtr cloud, float mr, float support_radius, vector<int>& key_indices);
void Harris3D_detector(PointCloudPtr cloud, float NMS_radius, vector<int>& key_indices);
void FPFH_descriptor(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float sup_radius, std::vector<std::vector<float>>& features);
void FPFH_descriptor(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, std::vector<int>& indices,
                     float sup_radius, std::vector<std::vector<float>>& features);
int Voxel_grid_downsample(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& new_cloud,
                      float leaf_size);
vector<int> removeInvalidPoint(PointCloudPtr cloud_in, vector<int>& keyPointIdx, float resolution);
/**********************************************3DCorres_methods***************************************/

pcl::PointCloud<pcl::PointXYZ>::Ptr getHarris3D_detector(PointCloudPtr cloud, float NMS_radius, vector<int>& key_indices);
PointCloudPtr removeInvalidkeyPoint(PointCloudPtr cloud_in, vector<int>& keyPointIdx, PointCloudPtr keyPoint, float resolution);
//=======================================
void boost_rand(int seed, int start, int end, int rand_num, std::vector<int>& idx);
void Rand_3(int seed, int scale, int& output1, int& output2, int& output3);
void RANSAC_trans_est(pcl::PointXYZ& point_s1, pcl::PointXYZ& point_s2, pcl::PointXYZ& point_s3,
	pcl::PointXYZ& point_t1, pcl::PointXYZ& point_t2, pcl::PointXYZ& point_t3, Eigen::Matrix4f& Mat);

/**********************************************Visualization***************************************/
void visualization(PointCloudPtr cloud_src, PointCloudPtr cloud_tar, /*PointCloudPtr keyPoint_src, PointCloudPtr keyPoint_tar,*/ Eigen::Matrix4d Mat, float resolution);
void visualization(PointCloudPtr &cloud_src, PointCloudPtr &cloud_tar, vector<Corre_3DMatch>&match, Eigen::Matrix4d &Mat, float &resolution);
void visualization(PointCloudPtr& ov_src, PointCloudPtr& cloud_src, PointCloudPtr& cloud_tar, vector<Corre_3DMatch>& match, Eigen::Matrix4d& Mat, Eigen::Matrix4d& GTmat, float& resolution);
int RANSAC(vector<Corre_3DMatch> Match, float resolution, int  _Iterations, Eigen::Matrix4f& Mat);
int RANSAC_score(vector<Corre_3DMatch> Match, float resolution, int  _Iterations, float threshold, Eigen::Matrix4f& Mat, string loss);
float Score_est(pcl::PointCloud<pcl::PointXYZ>::Ptr source_match_points, pcl::PointCloud<pcl::PointXYZ>::Ptr target_match_points, Eigen::Matrix4f Mat, float thresh, string loss);

void RMSE_visualization(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target, Eigen::Matrix4f& Mat_est, Eigen::Matrix4f& Mat_GT, float mr);
float RMSE_compute(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target, Eigen::Matrix4d& Mat_est, Eigen::Matrix4d& Mat_GT, float mr);
void cloud_viewer(PointCloudPtr cloud, const char* name);
void cloud_viewer_src_des(PointCloudPtr cloud_src, PointCloudPtr cloud_des);
void Corres_Viewer_Scorecolor(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_s, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_t, vector<Corre>& Hist_match, float& mr, int k);
void Corres_initial_visual(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_s, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_t, vector<Corre>& Hist_match, float& mr, Eigen::Matrix4d& GT_Mat);
void Corres_selected_visual(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_s, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_t, vector<Corre_3DMatch>& Hist_match, float& mr, float GT_thresh, Eigen::Matrix4d& GT_Mat);
void Corres_Viewer_Score(PointCloudPtr cloud_s, PointCloudPtr cloud_t, vector<Corre_3DMatch>& Hist_match, float& mr, int k);
bool compare_vote_score(const Vote& v1, const Vote& v2);
bool compare_vote_degree(const Vote_exp& v1, const Vote_exp& v2);
bool compare_corres_score(const Corre_3DMatch& c1, const Corre_3DMatch& c2);
vector<int> vectors_intersection(vector<int> v1, vector<int> v2);
double calculate_rotation_error(Eigen::Matrix3d& est, Eigen::Matrix3d& gt);
double calculate_translation_error(Eigen::Vector3d& est, Eigen::Vector3d& gt);
void sort_row(MatD& matrix, MatD& sorted_matrix, Eigen::MatrixXi& index);
void weight_SVD(PointCloudPtr& src_pts, PointCloudPtr& des_pts, Eigen::VectorXd& weights, double weight_threshold, Eigen::Matrix4d& trans_Mat);
double evaluation_trans(vector<Corre_3DMatch>& Match, vector<Corre_3DMatch>& correspondence, PointCloudPtr& src_corr_pts, PointCloudPtr& des_corr_pts, double weight_thresh, Eigen::Matrix4d& trans, double metric_thresh, const string &metric, float resolution, bool instance_equal);
bool evaluation_est(Eigen::Matrix4d est, Eigen::Matrix4d gt, double re_thresh, double te_thresh, double& RE, double& TE);
void print_and_destroy_cliques(igraph_vector_ptr_t* cliques);
void find_largest_clique_of_node(Eigen::MatrixXf& Graph, igraph_vector_ptr_t* cliques, vector<Corre_3DMatch>& correspondence, node_cliques* result, vector<int>& remain, int num_node, int est_num, string descriptor);
void post_refinement(vector<Corre_3DMatch>& correspondence, PointCloudPtr& src_corr_pts, PointCloudPtr& des_corr_pts, Eigen::Matrix4d& initial, double& best_score, double inlier_thresh, int iterations, const string &metric);
bool registration(const string &name,const string &src_pointcloud, const string &des_pointcloud,const string &corr_path, const string &label_path, const string &ov_label, const string &gt_mat, const string &folderPath, double& RE, double& TE, double& inlier_num, double& total_num, double& inlier_ratio, double& success_num, double& total_estimate, const string &descriptor, vector<double>& time_consumption);
bool registration(PointCloudPtr& src, PointCloudPtr& des, vector<Corre_3DMatch>& correspondence, vector<double>& ov_corr_label, string folderPath, float resolution, float cmp_thresh);
void GUO_ICP(PointCloudPtr& cloud_source, PointCloudPtr& cloud_target, float& mr, int& Max_iter_Num, Eigen::Matrix4f& Mat_ICP);
void sort_eigenvector(Eigen::VectorXd& eigenvector, Eigen::VectorXd& sorted_eigenvector, Eigen::VectorXi& index_eigenvector);
int GTM_corre_select(int Iterations, float mr, PointCloudPtr& cloud_source, PointCloudPtr& cloud_target, vector<Corre_3DMatch> Match, vector<int>& Match_inlier);
int Geometric_consistency(vector<Vote_exp>pts_degree, vector<int>& Match_inlier);
Eigen::VectorXf power_iteration(Eigen::MatrixXf& Graph, int iteration);
void savetxt(vector<Corre_3DMatch>corr, const string& save_path);
void computeCentroidAndCovariance(Corre_3DMatch& c, PointCloudPtr& src_knn, PointCloudPtr& des_knn);
#endif
