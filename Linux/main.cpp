#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include<iostream>
#include <string>
#include <numeric>
//#include<io.h>
#include<cstdlib>
#include <getopt.h>
#include "Eva.h"
#include <unistd.h>
using namespace std;
string folderPath;
bool add_overlap;
bool low_inlieratio;
bool no_logs;

string threeDMatch[8] = {
	"7-scenes-redkitchen",
	"sun3d-home_at-home_at_scan1_2013_jan_1",
	"sun3d-home_md-home_md_scan9_2012_sep_30",
	"sun3d-hotel_uc-scan3",
	"sun3d-hotel_umd-maryland_hotel1",
	"sun3d-hotel_umd-maryland_hotel3",
	"sun3d-mit_76_studyroom-76-1studyroom2",
	"sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika",
};

string threeDlomatch[8] = {
	"7-scenes-redkitchen_3dlomatch",
	"sun3d-home_at-home_at_scan1_2013_jan_1_3dlomatch",
	"sun3d-home_md-home_md_scan9_2012_sep_30_3dlomatch",
	"sun3d-hotel_uc-scan3_3dlomatch",
	"sun3d-hotel_umd-maryland_hotel1_3dlomatch",
	"sun3d-hotel_umd-maryland_hotel3_3dlomatch",
	"sun3d-mit_76_studyroom-76-1studyroom2_3dlomatch",
	"sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika_3dlomatch",
};

string ETH[4] = {
	"gazebo_summer",
	"gazebo_winter",
	"wood_autmn",
	"wood_summer",
};

double RE, TE, success_estimate_rate;
vector<int>scene_num;
vector<string> analyse(const string& name, const string& result_scene, const string& dataset_scene, const string& descriptor, ofstream& outfile, int iters, int data_index) {
	if (!no_logs && access(result_scene.c_str(), 0))
	{
		if (mkdir(result_scene.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)!=0) {
			cout << " Create scene folder failed! " << endl;
			exit(-1);
		}
	}
	vector<string>error_pair;

	string error_txt;
	//error_txt = result_scene + "/error_pair.txt";

	if (descriptor == "fpfh" || descriptor == "spinnet" || descriptor == "d3feat")
	{
		error_txt = dataset_scene + "/dataload.txt";
	}
	else if (descriptor == "fcgf")
	{
		error_txt = dataset_scene + "/dataload_fcgf.txt";
	}
	if (access(error_txt.c_str(), 0))
	{
		cout << " Could not find dataloader file! " << endl;
		exit(-1);
	}

	ifstream f1(error_txt);
	string line;
	while (getline(f1, line))
	{
		error_pair.push_back(line);
	}
	f1.close();
	scene_num.push_back(error_pair.size());
	vector<string>match_success_pair;
	int index = 1;
	RE = 0;
	TE = 0;
	success_estimate_rate = 0;
	vector<double>time;
	for (const auto& pair : error_pair)
	{
		time.clear();
		cout << "Pair " << index << ", " << "total " << error_pair.size() << " pairs." << endl;
		index++;
		string result_folder = result_scene + "/" + pair;
		string::size_type i = pair.find("+") + 1;
		string src_filename = dataset_scene + "/" + pair.substr(0, i - 1) + ".ply";
		string des_filename = dataset_scene + "/" + pair.substr(i, pair.length() - i) + ".ply";
		//cout << src_filename << " " << des_filename << endl;
		string corr_path = dataset_scene + "/" + pair + (descriptor == "fcgf" ? "@corr_fcgf.txt" : "@corr.txt");
		string gt_label = dataset_scene + "/" + pair + (descriptor == "fcgf" ? "@label_fcgf.txt" : "@label.txt");
		string gt_mat_path = dataset_scene + "/" + pair + (descriptor == "fcgf" ? "@GTmat_fcgf.txt" : "@GTmat.txt");

		string ov_label = "NULL";
		double re, te, inlier_num, total_num, inlier_ratio, success_estimate, total_estimate;
		int corrected = registration(name, src_filename, des_filename, corr_path, gt_label, ov_label, gt_mat_path, result_folder, re, te, inlier_num, total_num, inlier_ratio, success_estimate, total_estimate, descriptor, time);
		int iter = iters;
		double est_rr = success_estimate / (total_estimate / 1.0);
		success_estimate_rate += est_rr;
		if (corrected)
		{
			cout << pair << " Success." << endl;
			RE += re;
			TE += te;
			match_success_pair.push_back(pair);
		}
		else
		{
			cout << pair << " Fail." << endl;
		}
		outfile << pair << ',' << corrected << ',' << inlier_num << ',' << total_num << ',';
		outfile << setprecision(4) << inlier_ratio << ',' << est_rr << ',' << re << ',' << te << ',' << time[0] << ',' << time[1] << ',' << time[2] << ',' << time[3] << ',' << success_estimate << endl;
		cout << endl;
	}
	error_pair.clear();

	return match_success_pair;
}

void demo(){
    PointCloudPtr src_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    PointCloudPtr des_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    PointCloudPtr new_src_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    PointCloudPtr new_des_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    string src_path = "demo/src.pcd";
    string des_path = "demo/des.pcd";
    pcl::io::loadPCDFile(src_path, *src_cloud);
    pcl::io::loadPCDFile(des_path, *des_cloud);
    float src_resolution = MeshResolution_mr_compute(src_cloud);
    float des_resolution = MeshResolution_mr_compute(des_cloud);
    float resolution = (src_resolution + des_resolution) / 2;

    float downsample = 5 * resolution;
    Voxel_grid_downsample(src_cloud, new_src_cloud, downsample);
    Voxel_grid_downsample(des_cloud, new_des_cloud, downsample);
    vector<vector<float>> src_feature, des_feature;
    FPFH_descriptor(new_src_cloud, downsample*5, src_feature);
    FPFH_descriptor(new_des_cloud, downsample*5, des_feature);

    vector<Corre_3DMatch>correspondence;
    feature_matching(new_src_cloud, new_des_cloud, src_feature, des_feature, correspondence);

    vector<double>ov_lable;
    ov_lable.resize((int)correspondence.size());
    
    folderPath = "demo/result";
    cout << "Start registration." << endl;
    registration(src_cloud, des_cloud, correspondence, ov_lable, folderPath, resolution,0.99);
    //clear data
    src_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    des_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    new_src_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    new_des_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    src_feature.clear();
    src_feature.shrink_to_fit();
    des_feature.clear();
    des_feature.shrink_to_fit();
    correspondence.clear();
    correspondence.shrink_to_fit();
    ov_lable.clear();
    ov_lable.shrink_to_fit();
    exit(0);
}

void usage(){
    cout << "Usage:" << endl;
    cout << "\tHELP --help" <<endl;
    cout << "\tDEMO --demo" << endl;
    cout << "\tREQUIRED ARGS:" << endl;
    cout << "\t\t--output_path\toutput path for saving results." << endl;
    cout << "\t\t--input_path\tinput data path." << endl;
    cout << "\t\t--dataset_name\tdataset name. [3dmatch/3dlomatch/KITTI/ETH/U3M]" << endl;
    cout << "\t\t--descriptor\tdescriptor name. [fpfh/fcgf/spinnet/predator]" << endl;
    cout << "\t\t--start_index\tstart from given index. (begin from 0)" << endl;
    cout << "\tOPTIONAL ARGS:" << endl;
    cout << "\t\t--no_logs\tforbid generation of log files." << endl;
};

int main(int argc, char** argv) {
    //////////////////////////////////////////////////////////////////
    add_overlap = false;
    low_inlieratio = false;
    no_logs = false;
    int id = 0;
    string resultPath; 
    string datasetPath; 
    string datasetName; 
    string descriptor; 
    //////////////////////////////////////////////////////////////////
    int opt;
    int digit_opind = 0;
    int option_index = 0;
    static struct option long_options[] = {
            {"output_path", required_argument, NULL, 'o'},
            {"input_path", required_argument, NULL, 'i'},
            {"dataset_name", required_argument, NULL, 'n'},
            {"descriptor", required_argument, NULL, 'd'},
            {"start_index", required_argument, NULL, 's'},
            {"no_logs", optional_argument, NULL, 'g'},
            {"help", optional_argument, NULL, 'h'},
            {"demo", optional_argument, NULL, 'm'},
            {NULL, 0, 0, '\0'}
    };

    while((opt = getopt_long(argc, argv, "", long_options, &option_index)) != -1){
        switch (opt) {
            case 'h':
                usage();
                exit(0);
            case 'o':
                resultPath = optarg;
                break;
            case 'i':
                datasetPath = optarg;
                break;
            case 'n':
                datasetName = optarg;
                break;
            case 'd':
                descriptor = optarg;
                break;
            case 'g':
                no_logs = true;
                break;
            case 's':
                id = atoi(optarg);
                break;
            case 'm':
                demo();
                exit(0);
            case '?':
                printf("Unknown option: %c\n",(char)optopt);
                usage();
                exit(-1);
        }
    }
    if(argc  < 11){
        cout << 11 - argc <<" more args are required." << endl;
        usage();
        exit(-1);
    }

    cout << "Check your args setting:" << endl;
    cout << "\toutput_path: " << resultPath << endl;
    cout << "\tinput_path: " << datasetPath << endl;
    cout << "\tdataset_name: " << datasetName << endl;
    cout << "\tdescriptor: " << descriptor << endl;
    cout << "\tstart_index: " << id << endl;
    cout << "\tno_logs: " << no_logs << endl;

    sleep(5);

	int corrected = 0;
	int total_num = 0;
	double total_re = 0;
	double total_te = 0;
	vector<double>total_success_est_rate;
	vector<int> scene_correct_num;
	vector<double>scene_re_sum;
	vector<double>scene_te_sum;
	if (access(resultPath.c_str(), 0))
	{
		if (mkdir(resultPath.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
			cout << " Create save folder failed! " << endl;
			exit(-1);
		}
	}

	if (descriptor == "predator" && (datasetName == "3dmatch" || datasetName == "3dlomatch")) {
		vector<string>pairs;
		string loader = datasetPath + "/dataload.txt";
        cout << loader << endl;
		ifstream f1(loader);
		string line;
		while (getline(f1, line))
		{
			pairs.push_back(line);
		}
		f1.close();

		string analyse_csv = resultPath + "/" + datasetName + "_" + descriptor + ".csv";
		ofstream outFile;
		outFile.open(analyse_csv.c_str(), ios::out);
		outFile.setf(ios::fixed, ios::floatfield);
		outFile << "pair_name" << ',' << "corrected_or_no" << ',' << "inlier_num" << ',' << "total_num" << ',' << "inlier_ratio" << ',' << "RE" << ',' << "TE" << endl;
		vector<string>fail_pair;
		vector<double>time;
		for (int i = id; i < pairs.size(); i++)
		{
			time.clear();
			cout << "Pair " << i + 1 << "，total" << pairs.size()/*name_list.size()*/ << "，fail " << fail_pair.size() << endl;
			string filename = pairs[i];
			string corr_path = datasetPath + "/" + filename + "@corr.txt";
			string gt_mat_path = datasetPath + "/" + filename + "@GTmat.txt";
			string gt_label_path = datasetPath + "/" + filename + "@label.txt";
            string ov_label = "NULL";
			string folderPath = resultPath + "/" + filename;
			double re, te;
			double inlier_num, total_num;
			double inlier_ratio, success_estimate, total_estimate;

			int corrected = registration(datasetName, "NULL", "NULL", corr_path, gt_label_path, ov_label, gt_mat_path, folderPath, re, te, inlier_num, total_num, inlier_ratio, success_estimate, total_estimate, descriptor, time);
			if (corrected)
			{
				cout << filename << " Success." << endl;
				RE += re;
				TE += te;
			}
			else
			{
				fail_pair.push_back(filename);
				cout << filename << " Fail." << endl;
			}
			outFile << filename << ',' << corrected << ',' << inlier_num << ',' << total_num << ',';
			outFile << setprecision(4) << inlier_ratio << ',' << re << ',' << te << ',' << time[0] << ',' << time[1] << ',' << time[2] << ',' << time[3] << endl;
			cout << endl;
		}
		outFile.close();
		double success_num = pairs.size() - fail_pair.size();
		cout << "total:" << endl;
		cout << "\tRR:" << pairs.size() - fail_pair.size() << "/" << pairs.size() << " " << success_num / (pairs.size() / 1.0) << endl;
		cout << "\tRE:" << RE / (success_num / 1.0) << endl;
		cout << "\tTE:" << TE / (success_num / 1.0) << endl;
		cout << "fail pairs:" << endl;
		/*for (size_t i = 0; i < fail_pair.size(); i++)
		{
			cout << "\t" << fail_pair[i] << endl;
		}*/
	}
	else if (datasetName == "3dmatch")
	{
		for (size_t i = id; i < 8; i++)
		{
			cout << i + 1 << ":" << threeDMatch[i] << endl;
			string analyse_csv = resultPath + "/" + threeDMatch[i] + "_" + descriptor + ".csv";
			ofstream outFile;
			outFile.open(analyse_csv.c_str(), ios::out);
			outFile.setf(ios::fixed, ios::floatfield);
			outFile << "pair_name" << ',' << "corrected_or_no" << ',' << "inlier_num" << ',' << "total_num" << ',' << "inlier_ratio" << ',' << "est_rr" << ',' << "RE" << ',' << "TE" << ',' << "construction" << ',' << "search" << ',' << "selection" << ',' << "estimation" << endl;
			vector<string>matched = analyse("3dmatch", resultPath + "/" + threeDMatch[i], datasetPath + "/" + threeDMatch[i], descriptor, outFile, id, i);
			scene_re_sum.push_back(RE);
			scene_te_sum.push_back(TE);
			if (!matched.empty())
			{
				cout << endl;
				cout << threeDMatch[i] << ":" << endl;
				for (auto t : matched)
				{
					cout << "\t" << t << endl;
				}
				cout << endl;
				cout << threeDMatch[i] << ":" << matched.size() / (scene_num[i] / 1.0) << endl;
				cout << "success_est_rate:" << success_estimate_rate / (scene_num[i] / 1.0) << "RE:" << RE / matched.size() << "\tTE:" << TE / matched.size() << endl;
				corrected += matched.size();
				total_success_est_rate.push_back(success_estimate_rate);
				scene_correct_num.push_back(matched.size());
			}
			outFile.close();
			matched.clear();
		}
		string detail_txt = resultPath + "/details.txt";
		ofstream outFile;
		outFile.open(detail_txt.c_str(), ios::out);
		outFile.setf(ios::fixed, ios::floatfield);
		for (size_t i = 0; i < 8; i++)
		{
			total_num += scene_num[i];
			total_re += scene_re_sum[i];
			total_te += scene_te_sum[i];
			cout << i + 1 << ":" << endl;
			outFile << i + 1 << ":" << endl;
			cout << "\tRR: " << scene_correct_num[i] << "/" << scene_num[i] << " " << scene_correct_num[i] / (scene_num[i] / 1.0) << endl;
			outFile << "\tRR: " << scene_correct_num[i] << "/" << scene_num[i] << " " << setprecision(4) << scene_correct_num[i] / (scene_num[i] / 1.0) << endl;
			cout << "\tSuccess_est_rate: " << total_success_est_rate[i] / (scene_num[i] / 1.0) << endl;
			cout << "\tRE: " << scene_re_sum[i] / (scene_correct_num[i] / 1.0) << endl;
			outFile << "\tRE: " << setprecision(4) << scene_re_sum[i] / (scene_correct_num[i] / 1.0) << endl;
			cout << "\tTE: " << scene_te_sum[i] / (scene_correct_num[i] / 1.0) << endl;
			outFile << "\tTE: " << setprecision(4) << scene_te_sum[i] / (scene_correct_num[i] / 1.0) << endl;
		}
		cout << "total:" << endl;
		outFile << "total:" << endl;
		cout << "\tRR: " << corrected / (total_num / 1.0) << endl;
		outFile << "\tRR: " << setprecision(4) << corrected / (total_num / 1.0) << endl;
		cout << "\tSuccess_est_rate: " << accumulate(total_success_est_rate.begin(), total_success_est_rate.end(), 0.0) / (total_num / 1.0) << endl;
		cout << "\tRE: " << total_re / (corrected / 1.0) << endl;
		outFile << "\tRE: " << setprecision(4) << total_re / (corrected / 1.0) << endl;
		cout << "\tTE: " << total_te / (corrected / 1.0) << endl;
		outFile << "\tTE: " << setprecision(4) << total_te / (corrected / 1.0) << endl;
		outFile.close();
	}
	else if (datasetName == "3dlomatch")
	{
		for (size_t i = id; i < 8; i++)
		{
			string analyse_csv = resultPath + "/" + threeDlomatch[i] + "_" + descriptor + ".csv";
			ofstream outFile;
			outFile.open(analyse_csv.c_str(), ios::out);
			outFile.setf(ios::fixed, ios::floatfield);
			outFile << "pair_name" << ',' << "corrected_or_no" << ',' << "inlier_num" << ',' << "total_num" << ',' << "inlier_ratio" << ',' << "est_rr" << ',' << "RE" << ',' << "TE" << endl;
			vector<string>matched = analyse("3dlomatch", resultPath + "/" + threeDlomatch[i], datasetPath + "/" + threeDlomatch[i], descriptor, outFile, id, i);
			scene_re_sum.push_back(RE);
			scene_te_sum.push_back(TE);
			if (!matched.empty())
			{
				cout << endl;
				cout << threeDlomatch[i] << ":" << endl;
				for (auto t : matched)
				{
					cout << "\t" << t << endl;
				}
				cout << endl;
				cout << threeDlomatch[i] << ":" << matched.size() / (scene_num[i] / 1.0) << endl;
				cout << "RE:" << RE / matched.size() << "\tTE:" << TE / matched.size() << endl;
				corrected += matched.size();
				total_success_est_rate.push_back(success_estimate_rate);
				scene_correct_num.push_back(matched.size());
			}
			outFile.close();
			matched.clear();
		}
		string detail_txt = resultPath + "/details.txt";
		ofstream outFile;
		outFile.open(detail_txt.c_str(), ios::out);
		outFile.setf(ios::fixed, ios::floatfield);
		for (size_t i = 0; i < 8; i++)
		{
			total_num += scene_num[i];
			total_re += scene_re_sum[i];
			total_te += scene_te_sum[i];
			cout << i + 1 << ":" << endl;
			outFile << i + 1 << ":" << endl;
			cout << "\tRR: " << scene_correct_num[i] << "/" << scene_num[i] << " " << scene_correct_num[i] / (scene_num[i] / 1.0) << endl;
			outFile << "\tRR: " << scene_correct_num[i] << "/" << scene_num[i] << " " << setprecision(4) << scene_correct_num[i] / (scene_num[i] / 1.0) << endl;
			cout << "\tSuccess_est_rate: " << total_success_est_rate[i] / (scene_num[i] / 1.0) << endl;
			cout << "\tRE: " << scene_re_sum[i] / (scene_correct_num[i] / 1.0) << endl;
			outFile << "\tRE: " << setprecision(4) << scene_re_sum[i] / (scene_correct_num[i] / 1.0) << endl;
			cout << "\tTE: " << scene_te_sum[i] / (scene_correct_num[i] / 1.0) << endl;
			outFile << "\tTE: " << setprecision(4) << scene_te_sum[i] / (scene_correct_num[i] / 1.0) << endl;
		}
		cout << "total:" << endl;
		outFile << "total:" << endl;
		cout << "\tRR: " << corrected / (total_num / 1.0) << endl;
		outFile << "\tRR: " << setprecision(4) << corrected / (total_num / 1.0) << endl;
		cout << "\tSuccess_est_rate: " << accumulate(total_success_est_rate.begin(), total_success_est_rate.end(), 0.0) / (total_num / 1.0) << endl;
		cout << "\tRE: " << total_re / (corrected / 1.0) << endl;
		outFile << "\tRE: " << setprecision(4) << total_re / (corrected / 1.0) << endl;
		cout << "\tTE: " << total_te / (corrected / 1.0) << endl;
		outFile << "\tTE: " << setprecision(4) << total_te / (corrected / 1.0) << endl;
		outFile.close();
	}
	else if (datasetName == "ETH")
	{
		for (size_t i = id; i < 4; i++)
		{
			cout << i + 1 << ":" << ETH[i] << endl;
			string analyse_csv = resultPath + "/" + ETH[i] + "_" + descriptor + ".csv";
			ofstream outFile;
			outFile.open(analyse_csv.c_str(), ios::out);
			outFile.setf(ios::fixed, ios::floatfield);
			outFile << "pair_name" << ',' << "corrected_or_no" << ',' << "inlier_num" << ',' << "total_num" << ',' << "inlier_ratio" << ',' << "est_rr" << ',' << "RE" << ',' << "TE" << ',' << "construction" << ',' << "search" << ',' << "selection" << ',' << "estimation" << endl;
			vector<string>matched = analyse("3dmatch", resultPath + "/" + ETH[i], datasetPath + "/" + ETH[i], descriptor, outFile, id, i);
			scene_re_sum.push_back(RE);
			scene_te_sum.push_back(TE);
			if (!matched.empty())
			{
				cout << endl;
				cout << ETH[i] << ":" << endl;
				for (auto t : matched)
				{
					cout << "\t" << t << endl;
				}
				cout << endl;
				cout << ETH[i] << ":" << matched.size() << endl;
				cout << "success_est_rate:" << success_estimate_rate / (scene_num[i] / 1.0) << "RE:" << RE / matched.size() << "\tTE:" << TE / matched.size() << endl;
				corrected += matched.size();
				total_success_est_rate.push_back(success_estimate_rate);
				scene_correct_num.push_back(matched.size());
			}
			outFile.close();
			matched.clear();
		}
		string detail_txt = resultPath + "/details.txt";
		ofstream outFile;
		outFile.open(detail_txt.c_str(), ios::out);
		outFile.setf(ios::fixed, ios::floatfield);
		for (size_t i = 0; i < 4; i++)
		{
			total_num += scene_num[i];
			total_re += scene_re_sum[i];
			total_te += scene_te_sum[i];
			cout << i + 1 << ":" << endl;
			outFile << i + 1 << ":" << endl;
			cout << "\tRR: " << scene_correct_num[i] << "/" << scene_num[i] << " " << scene_correct_num[i] / (scene_num[i] / 1.0) << endl;
			outFile << "\tRR: " << scene_correct_num[i] << "/" << scene_num[i] << " " << setprecision(4) << scene_correct_num[i] / (scene_num[i] / 1.0) << endl;
			cout << "\tSuccess_est_rate: " << total_success_est_rate[i] / (scene_num[i] / 1.0) << endl;
			cout << "\tRE: " << scene_re_sum[i] / (scene_correct_num[i] / 1.0) << endl;
			outFile << "\tRE: " << setprecision(4) << scene_re_sum[i] / (scene_correct_num[i] / 1.0) << endl;
			cout << "\tTE: " << scene_te_sum[i] / (scene_correct_num[i] / 1.0) << endl;
			outFile << "\tTE: " << setprecision(4) << scene_te_sum[i] / (scene_correct_num[i] / 1.0) << endl;
		}
		cout << "total:" << endl;
		outFile << "total:" << endl;
		cout << "\tRR: " << corrected / (total_num / 1.0) << endl;
		outFile << "\tRR: " << setprecision(4) << corrected / (total_num / 1.0) << endl;
		cout << "\tSuccess_est_rate: " << accumulate(total_success_est_rate.begin(), total_success_est_rate.end(), 0.0) / (total_num / 1.0) << endl;
		cout << "\tRE: " << total_re / (corrected / 1.0) << endl;
		outFile << "\tRE: " << setprecision(4) << total_re / (corrected / 1.0) << endl;
		cout << "\tTE: " << total_te / (corrected / 1.0) << endl;
		outFile << "\tTE: " << setprecision(4) << total_te / (corrected / 1.0) << endl;
		outFile.close();
	}
	else if (datasetName == "KITTI")
	{
		int pair_num = 555;
		//string txt_path = datasetPath + "/" + descriptor;
        const string& txt_path = datasetPath;
		string analyse_csv = resultPath + "/KITTI_" + descriptor + ".csv";
		ofstream outFile;
		outFile.open(analyse_csv.c_str(), ios::out);
		outFile.setf(ios::fixed, ios::floatfield);
		outFile << "pair_name" << ',' << "corrected_or_no" << ',' << "inlier_num" << ',' << "total_num" << ',' << "inlier_ratio" << ',' << "RE" << ',' << "TE" << endl;
		vector<string>fail_pair;
		vector<double>time;
		for (int i = id; i < pair_num; i++)
		{
			time.clear();
			cout << "Pair " << i + 1 << "，total" << pair_num/*name_list.size()*/ << "，fail " << fail_pair.size() << endl;

            string filename = to_string(i);/*name_list[i]*/;
			string corr_path = txt_path + "/" + filename + '/' + descriptor + "@corr.txt";
			string gt_mat_path = txt_path + "/" + filename + '/' + descriptor + "@gtmat.txt";
			string gt_label_path = txt_path + "/" + filename + '/' + descriptor + "@gtlabel.txt";
            string ov_label = "NULL";
			string folderPath = resultPath + "/" + filename;
			double re, te;
			double inlier_num, total_num;
			double inlier_ratio, success_estimate, total_estimate;

			int corrected = registration("KITTI", "NULL", "NULL", corr_path, gt_label_path, ov_label, gt_mat_path, folderPath, re, te, inlier_num, total_num, inlier_ratio, success_estimate, total_estimate, descriptor, time);
			if (corrected)
			{
				cout << filename << " Success." << endl;
				RE += re;
				TE += te;
			}
			else
			{
				fail_pair.push_back(filename);
				cout << filename << " Fail." << endl;
			}
			outFile << filename << ',' << corrected << ',' << inlier_num << ',' << total_num << ',';
			outFile << setprecision(4) << inlier_ratio << ',' << re << ',' << te << endl;
			cout << endl;
		}
		outFile.close();
		double success_num = pair_num - fail_pair.size();
		cout << "total:" << endl;
		cout << "\tRR:" << pair_num - fail_pair.size() << "/" << pair_num << " " << success_num / (pair_num / 1.0) << endl;
		cout << "\tRE:" << RE / (success_num / 1.0) << endl;
		cout << "\tTE:" << TE / (success_num / 1.0) << endl;
		cout << "fail pairs:" << endl;
		for (size_t i = 0; i < fail_pair.size(); i++)
		{
			cout << "\t" << fail_pair[i] << endl;
		}
	}
	else if (datasetName == "U3M")
	{
		//预处理
		vector<string>pairname;
		string dataload_path = datasetPath + "/dataload.txt";
		FILE* fp = fopen(dataload_path.c_str(), "r");
		while (!feof(fp)) {
			char src[1024], des[1024];
			Eigen::Matrix4d GTmat;
			fscanf(fp, "%s %s", &src, &des);
			for (int i = 0; i < 16; i++) {
				fscanf(fp, " %lf", &GTmat(i / 4, i % 4));
			}
			fscanf(fp, "\n");
			string srcfile(src);
			string desfile(des);
			string folderName = srcfile + "+" + desfile;
			string folderPath = resultPath + "/" + folderName;

			if (access(folderPath.c_str(), 0))
			{
				if (mkdir(folderPath.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
					cout << " 创建文件夹 " << folderName << " 失败 " << endl;
					exit(-1);
				}
			}

			string srcPath = datasetPath + "/" + srcfile;
			string desPath = datasetPath + "/" + desfile;
			cout << srcPath << " " << desPath << endl;
			string gtmatPath = folderPath + "/GTmat.txt";
			string corrPath = folderPath + "/corr.txt";
			string labelPath = folderPath + "/true_corre.txt";
//			PointCloudPtr cloud_src(new pcl::PointCloud<pcl::PointXYZ>);
//			XYZorPly_Read(srcPath, cloud_src);
//			PointCloudPtr cloud_des(new pcl::PointCloud<pcl::PointXYZ>);
//			XYZorPly_Read(desPath, cloud_des);

			//计算点云分辨率
//			float resolution_src = MeshResolution_mr_compute(cloud_src);
//			float resolution_des = MeshResolution_mr_compute(cloud_des);
//			float mr = (resolution_des + resolution_src) / 2;
//			float sup_radius = 15 * mr; //15
//			float NMS_radius = 2.7 * mr; //2.7
//			float GT_thresh = 5 * mr;
//			vector<int>Idx_model, Idx_scene;
//			pcl::PointCloud<pcl::PointXYZ>::Ptr keypoint_src;
//			pcl::PointCloud<pcl::PointXYZ>::Ptr keypoint_tar;
//			keypoint_src = getHarris3D_detector(cloud_src, NMS_radius, Idx_model);
//			keypoint_tar = getHarris3D_detector(cloud_des, NMS_radius, Idx_scene);
//			keypoint_src = removeInvalidkeyPoint(cloud_src, Idx_model, keypoint_src, mr);
//			keypoint_tar = removeInvalidkeyPoint(cloud_des, Idx_scene, keypoint_tar, mr);
//			vector<vector<float>> f_s, f_t;
//			vector<LRF> LRFs_s, LRFs_t;
//			SHOT_compute(cloud_src, Idx_model, sup_radius, f_s, LRFs_s);
//			SHOT_compute(cloud_des, Idx_scene, sup_radius, f_t, LRFs_t);
//			vector<Corre> Corres_initial;
//			feature_matching(cloud_src, cloud_des, LRFs_s, LRFs_t, Idx_model, Idx_scene, f_s, f_t, Corres_initial);
//			int true_correct_num = Correct_corre_compute(cloud_src, cloud_des, Corres_initial, GT_thresh, GTmat, folderPath);
//			if (true_correct_num == 0)
//			{
//				cout << " 没有正确匹配! " << endl;
//				//exit(0);
//			}
//			if (true_correct_num >= 3)
//			{
//				pairname.push_back(folderName);
//			}
//			cout << true_correct_num << "/" << Corres_initial.size() << endl;
//			FILE* fp1 = fopen(corrPath.c_str(), "w");
//			for (size_t i = 0; i < Corres_initial.size(); i++)
//			{
//				fprintf(fp1, "%f %f %f %f %f %f\n", cloud_src->points[Corres_initial[i].source_idx].x, cloud_src->points[Corres_initial[i].source_idx].y, cloud_src->points[Corres_initial[i].source_idx].z, cloud_des->points[Corres_initial[i].target_idx].x, cloud_des->points[Corres_initial[i].target_idx].y, cloud_des->points[Corres_initial[i].target_idx].z);
//			}
//			fclose(fp1);
//			FILE* fp2 = fopen(gtmatPath.c_str(), "w");
//			for (int i = 0; i < 4; i++)
//			{
//				for (int j = 0; j < 3; j++) {
//					fprintf(fp2, "%lf ", GTmat(i, j));
//				}
//				fprintf(fp2, "%lf\n", GTmat(i, 3));
//			}
//			fclose(fp2);
            //overlap label
//            if(add_overlap){
//                string ov_label = folderPath + "/ov.txt";
//                PointCloudPtr src_trans(new pcl::PointCloud<pcl::PointXYZ>);
//                pcl::transformPointCloud(*cloud_src, *src_trans, GTmat);
//                pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_src_trans, kdtree_des;
//                kdtree_src_trans.setInputCloud(src_trans);
//                kdtree_des.setInputCloud(cloud_des);
//                vector<int>src_ind(1), des_ind(1);
//                vector<float>src_dis(1), des_dis(1);
//                fp1 = fopen(ov_label.c_str(), "w");
//                for(auto & i : Corres_initial){
//                    pcl::PointXYZ src_query = src_trans->points[i.source_idx];
//                    pcl::PointXYZ des_query = cloud_des->points[i.target_idx];
//                    kdtree_des.nearestKSearch(src_query, 1, des_ind, src_dis);
//                    kdtree_src_trans.nearestKSearch(des_query, 1, src_ind, des_dis);
//                    int src_ov_score = src_dis[0] > GT_thresh ? 0 : 1; //square dist  <= GT_thresh
//                    int des_ov_score = des_dis[0] > GT_thresh ? 0 : 1;
//                    if(src_ov_score && des_ov_score){
//                        fprintf(fp1, "1\n");
//                    }
//                    else{
//                        fprintf(fp1, "0\n");
//                    }
//                }
//                fclose(fp1);
//            }
            FILE *fp1 = fopen(labelPath.c_str(), "r");
            int cnt = 0;
            while(!feof(fp1))
			{
                int value;
                fscanf(fp1, "%d\n", &value);
                cnt += value;
                if(cnt == 3){
                    pairname.push_back(folderName);
                    break;
                }
			}
            fclose(fp1);
		}
		fclose(fp);
		cout << "pre-process finish" << endl;
		double RMSE_threshold[10] = { 0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0 };
		double RR[10] = { 0 };
		int pairnum = pairname.size();
        cout << pairnum << endl;
		string analyse_csv = resultPath + "/U3M.csv";
		ofstream outFile;
		outFile.open(analyse_csv.c_str(), ios::out);
		outFile.setf(ios::fixed, ios::floatfield);
		outFile << "pair_name" << ',' << "corrected_or_no" << ',' << "inlier_num" << ',' << "total_num" << ',' << "inlier_ratio" << endl;
		vector<double>time;
		for (int j = 0; j < pairnum; j++)
		{
			time.clear();
			cout << "Pair " << j + 1 << "，total " << pairnum/*name_list.size()*/ << endl;
			string filename = pairname[j];
			string::size_type pos = filename.find_last_of('+') + 1;
			string src_pointcloud = datasetPath + "/" + filename.substr(0, pos - 1);
			string des_pointcloud = datasetPath + "/" + filename.substr(pos, filename.length() - pos);
			string folderPath = resultPath + "/" + filename;
			string corr_path = folderPath + "/corr.txt";
			string gt_mat_path = folderPath + "/GTmat.txt";
			string gt_label_path = folderPath + "/true_corre.txt";
			double re, te;
			double inlier_num, total_num;
			double inlier_ratio, success_estimate, total_estimate;
			string ov_label = "NULL";
			int corrected = registration("U3M", src_pointcloud, des_pointcloud, corr_path, gt_label_path, ov_label, gt_mat_path, folderPath, re, te, inlier_num, total_num, inlier_ratio, success_estimate, total_estimate, descriptor, time);
			cout << filename;
			for (int i = 0; i < 10; i++)
			{
				if (re <= RMSE_threshold[i])
				{
					RR[i] ++;
					cout << " 1";
				}
				else {
					cout << " 0";
				}
			}
			outFile << filename << ',' << corrected << ',' << inlier_num << ',' << total_num << ',';
			outFile << setprecision(4) << inlier_ratio << endl;
			cout << endl;
			cout << endl;
		}
		outFile.close();
		for (double i : RR)
		{
			cout << i / (pairnum / 1.0) << " " << endl;
		}
	}
	else {
		exit(0);
	}
    return 0;
}
