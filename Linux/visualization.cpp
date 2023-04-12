#include <stdio.h>
#include <vector>
#include <time.h>
#include <algorithm>
#include <pcl/point_types.h>
#include <pcl/registration/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <thread>
#include <pcl/visualization/pcl_visualizer.h>
#include "Eva.h"
void prseodu_color(float max_value, vector<float>scores, vector<Vertex>& RGB)
{
	RGB.resize(scores.size());
	for (int i = 0; i < scores.size(); i++)
	{

		float r, g, b;
		float val = scores[i] * 255 / max_value;
		//red  
		if (val < 128)
		{
			r = 0;
		}
		else if (val < 192)
		{
			r = 255.0f / 64 * (val - 128);
		}
		else
		{
			r = 255.0f;
		}
		//green  
		if (val < 64)
		{
			g = 255.0 / 64 * val;
		}
		else if (val < 192)
		{
			g = 255.0;
		}
		else
		{
			g = -255.0 / 63 * (val - 192) + 255;
		}
		//blue  
		if (val < 64)
		{
			b = 255.0;
		}
		else if (val < 128)
		{
			b = -255.0 / 63 * (val - 64) + 255;
		}
		else
		{
			b = 0.0;
		}
		Vertex temp;
		temp.x = r; temp.y = g; temp.z = b;
		RGB[i] = temp;
	}
}
boost::shared_ptr<pcl::visualization::PCLVisualizer> rgbVis(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(255, 255, 255);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
	//viewer->addCoordinateSystem (1.0);
	viewer->initCameraParameters();
	return (viewer);
}
void cloud_viewer_RGB(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, vector<Vertex>colors)
{
	int i;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr result(new pcl::PointCloud<pcl::PointXYZRGB>);
	result->width = cloud->size();
	result->height = 1;
	result->is_dense = true;
	result->points.resize(result->width * result->height);
	///////////////////////////////////////////////////////
	for (i = 0; i < cloud->size(); i++)
	{
		uint8_t r(colors[i].x), g = (colors[i].y), b = (colors[i].z);
		uint32_t rgb = (static_cast<uint32_t>(r) << 16 |
			static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
		result->points[i].x = cloud->points[i].x;
		result->points[i].y = cloud->points[i].y;
		result->points[i].z = cloud->points[i].z;
		result->points[i].rgb = *reinterpret_cast<float*>(&rgb);
	}
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_rgb;
	viewer_rgb = rgbVis(result);
	while (!viewer_rgb->wasStopped())
	{
		viewer_rgb->spinOnce();
	}
}
bool next_iteration = false;
void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event, void* nothing)
{
	if (event.getKeySym() == "space" && event.keyDown())
		next_iteration = true;
}
//point cloud visualization
void cloud_viewer(PointCloudPtr cloud, const char* name)
{
	pcl::visualization::PCLVisualizer viewer(name);
	pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_src(cloud, 220, 20, 60);
	pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_tar(cloud, 30, 144, 255);
	if (name == "src")viewer.addPointCloud(cloud, cloud_color_handler_src, name);
	if (name == "tar")viewer.addPointCloud(cloud, cloud_color_handler_tar, name);
	viewer.setBackgroundColor(255, 255, 255);
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}

}

void cloud_viewer_src_des(PointCloudPtr cloud_src, PointCloudPtr cloud_des) {
	pcl::visualization::PCLVisualizer viewer("src and des");
	int v1(0);
	int v2(1);
	// 定义显示的颜色信息
	float bckgr_gray_level = 0.0;  // 黑色
	float txt_gray_lvl = 1.0 - bckgr_gray_level;
	viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_src_color(cloud_src, 0, 0, 255);
	viewer.addPointCloud(cloud_src, cloud_src_color, "cloud_src_v1", v1);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_des_color(cloud_des, 255, 0, 0);
	viewer.addPointCloud(cloud_des, cloud_des_color, "cloud_des_v2", v2);
	// 加入文本的描述在各自的视口界面
	viewer.addText("src point cloud\n", 10, 15, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "info_1", v1);
	viewer.addText("des point cloud\n", 10, 15, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "info_2", v2);
	//设置背景颜色
	viewer.setBackgroundColor(bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v1);
	viewer.setBackgroundColor(bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v2);

	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}

//registration result
void visualization(PointCloudPtr cloud_src, PointCloudPtr cloud_tar, /*PointCloudPtr keyPoint_src, PointCloudPtr keyPoint_tar,*/ Eigen::Matrix4d Mat, float resolution)
{
	//cout << "visu resolution=" << resolution << endl;
	//visulization
	pcl::visualization::PCLVisualizer viewer("Registration");
	pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_src(cloud_src, sR, sG, sB);
	pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_tar(cloud_tar, tR, tG, tB);
	//pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> keyPoint_color_handler_src(keyPoint_src, 0, 255, 0);
	//pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> keyPoint_color_handler_tar(keyPoint_tar, 255, 185, 15);
	//Add pointcloud
	viewer.addPointCloud(cloud_src, cloud_color_handler_src, "cloud_src");
	viewer.addPointCloud(cloud_tar, cloud_color_handler_tar, "cloud_tar");
	//Add keyPoint
	//viewer.addPointCloud(keyPoint_src, keyPoint_color_handler_src, "keyPoint_src");
	//viewer.addPointCloud(keyPoint_tar, keyPoint_color_handler_tar, "keyPoint_tar");
	//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "keyPoint_src");
	//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "keyPoint_tar");
	//for (size_t j = 0; j < match.size(); ++j)
	//{
	//	int idx1 = match[j].source_idx;
	//	int idx2 = match[j].target_idx;
	//	std::stringstream SS_line_b;
	//	SS_line_b << "line" << j;
	//	viewer.addLine< pcl::PointXYZ, pcl::PointXYZ>(cloud_src->points[idx1], cloud_tar->points[idx2], 0, 255, 0, SS_line_b.str());
	//}
	//viewer.addSphere<pcl::PointXYZ>(cloud_src->points[500], 5 * resolution, "sphere", 0);
	viewer.setBackgroundColor(0, 0, 0);
    viewer.addText("Before registration\n", 10, 15, 20, 255, 255, 255, "info_1");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_src");

	// Set camera position and orientation
	//viewer.setCameraPosition(-3.68332, 2.94092, 5.71266, 0.289847, 0.921947, -0.256907, 0);
	//viewer.setSize(1280, 1024);
	//transform
	viewer.registerKeyboardCallback(&keyboardEventOccurred, (void*)NULL);
    cout << "Press space to register." << endl;
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
		//transform
		if (next_iteration)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trans_src(new pcl::PointCloud<pcl::PointXYZ>);
			viewer.removeAllShapes();
			viewer.removePointCloud("cloud_src");
            viewer.updateText("After registration\n", 10, 15, 20, 255, 255, 255, "info_1");
			pcl::transformPointCloud(*cloud_src, *cloud_trans_src, Mat);
			viewer.addPointCloud(cloud_trans_src, cloud_color_handler_src, "cloud_trans_src");
			//viewer.removePointCloud("keyPoint_src");
			//viewer.removePointCloud("keyPoint_tar");
			//viewer.addSphere<pcl::PointXYZ>(cloud_src->points[500], 5 * resolution, "sphere", 0);
		}
		next_iteration = false;
	}
    this_thread::sleep_for(100ms);
	//system("pause");
}
//point-wise ijumerror_map
void RMSE_visualization(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target, Eigen::Matrix4f& Mat_est, Eigen::Matrix4f& Mat_GT, float mr)
{
	int i;
	float upper_bound = 10.0f;
	vector<Vertex>RGB;
	for (i = 0; i < cloud_source->points.size(); i++)
	{
		Vertex temp = { 135,135,135 };
		RGB.push_back(temp);
	}
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source_trans_GT(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*cloud_source, *cloud_source_trans_GT, Mat_GT);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source_trans_EST(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*cloud_source, *cloud_source_trans_EST, Mat_est);
	vector<int>overlap_idx; float overlap_thresh = 4 * mr;
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree1;
	pcl::PointXYZ query_point;
	vector<int>pointIdx;
	vector<float>pointDst;
	kdtree1.setInputCloud(cloud_target);
	for (i = 0; i < cloud_source_trans_GT->points.size(); i++)
	{
		query_point = cloud_source_trans_GT->points[i];
		kdtree1.nearestKSearch(query_point, 1, pointIdx, pointDst);
		if (sqrt(pointDst[0]) <= overlap_thresh)
			overlap_idx.push_back(i);
	}
	//
	vector<float>errors;
	for (i = 0; i < overlap_idx.size(); i++)
	{
		query_point = cloud_source_trans_EST->points[overlap_idx[i]];
		float dist_x = pow(cloud_source_trans_EST->points[overlap_idx[i]].x - cloud_source_trans_GT->points[overlap_idx[i]].x, 2);
		float dist_y = pow(cloud_source_trans_EST->points[overlap_idx[i]].y - cloud_source_trans_GT->points[overlap_idx[i]].y, 2);
		float dist_z = pow(cloud_source_trans_EST->points[overlap_idx[i]].z - cloud_source_trans_GT->points[overlap_idx[i]].z, 2);
		float dist = sqrt(dist_x + dist_y + dist_z) / mr;
		errors.push_back(dist);
	}
	vector<Vertex>RGB_overlap;
	prseodu_color(upper_bound, errors, RGB_overlap);
	for (i = 0; i < RGB_overlap.size(); i++)
	{
		RGB[overlap_idx[i]] = RGB_overlap[i];
	}
	cloud_viewer_RGB(cloud_source, RGB);
}
float RMSE_compute(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target, Eigen::Matrix4d& Mat_est, Eigen::Matrix4d& Mat_GT, float mr)
{
	float RMSE_temp = 0.0f;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source_trans_GT(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*cloud_source, *cloud_source_trans_GT, Mat_GT);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source_trans_EST(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*cloud_source, *cloud_source_trans_EST, Mat_est);
	vector<int>overlap_idx; float overlap_thresh = 4 * mr;
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree1;
	pcl::PointXYZ query_point;
	vector<int>pointIdx;
	vector<float>pointDst;
	kdtree1.setInputCloud(cloud_target);
	for (int i = 0; i < cloud_source_trans_GT->points.size(); i++)
	{
		query_point = cloud_source_trans_GT->points[i];
		kdtree1.nearestKSearch(query_point, 1, pointIdx, pointDst);
		if (sqrt(pointDst[0]) <= overlap_thresh)
			overlap_idx.push_back(i);
	}
	//
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree2;
	kdtree2.setInputCloud(cloud_source_trans_GT);
	for (int i = 0; i < overlap_idx.size(); i++)
	{
		//query_point = cloud_source_trans_EST->points[overlap_idx[i]];
		//kdtree2.nearestKSearch(query_point,1,pointIdx,pointDst); RMSE_temp+=sqrt(pointDst[0]);
		float dist_x = pow(cloud_source_trans_EST->points[overlap_idx[i]].x - cloud_source_trans_GT->points[overlap_idx[i]].x, 2);
		float dist_y = pow(cloud_source_trans_EST->points[overlap_idx[i]].y - cloud_source_trans_GT->points[overlap_idx[i]].y, 2);
		float dist_z = pow(cloud_source_trans_EST->points[overlap_idx[i]].z - cloud_source_trans_GT->points[overlap_idx[i]].z, 2);
		float dist = sqrt(dist_x + dist_y + dist_z);
		RMSE_temp += dist;
	}
	RMSE_temp /= overlap_idx.size();
	RMSE_temp /= mr;
	//
	return RMSE_temp;
}

void Corres_Viewer_Score(PointCloudPtr cloud_s, PointCloudPtr cloud_t, vector<Corre_3DMatch>& Hist_match, float& mr, int k) {
	int i;
	float gap = Corres_view_gap * mr;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_add(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointXYZRGB tar_temp;
	uint8_t r1(sR), g1 = (sG), b1 = (sB), r2(tR), g2 = (tG), b2 = (tB);
	uint32_t rgb1 = (static_cast<uint32_t>(r1) << 16 |
		static_cast<uint32_t>(g1) << 8 | static_cast<uint32_t>(b1));
	uint32_t rgb2 = (static_cast<uint32_t>(r2) << 16 |
		static_cast<uint32_t>(g2) << 8 | static_cast<uint32_t>(b2));
	for (i = 0; i < cloud_s->points.size(); i++)
	{
		tar_temp.x = cloud_s->points[i].x;
		tar_temp.y = cloud_s->points[i].y;
		tar_temp.z = cloud_s->points[i].z;
		tar_temp.rgb = *reinterpret_cast<float*>(&rgb1);
		cloud_add->points.push_back(tar_temp);
	}
	for (i = 0; i < cloud_t->points.size(); i++)
	{

		tar_temp.x = cloud_t->points[i].x - Corres_view_gap * 0.25 * mr;
		//tar_temp.y = cloud_t->points[i].y + gap;
		tar_temp.y = cloud_t->points[i].y + Corres_view_gap * 1 * mr;
		tar_temp.z = cloud_t->points[i].z;
		tar_temp.rgb = *reinterpret_cast<float*>(&rgb2);

		cloud_add->points.push_back(tar_temp);
	}
	float center_x = 0, center_y = 0, center_z = 0;
	for (i = 0; i < cloud_add->points.size(); i++)
	{
		center_x += cloud_add->points[i].x; center_y += cloud_add->points[i].y; center_z += cloud_add->points[i].z;
	}
	center_x /= cloud_add->points.size(); center_y /= cloud_add->points.size(); center_z /= cloud_add->points.size();
	for (i = 0; i < cloud_add->points.size(); i++)
	{
		cloud_add->points[i].x -= center_x; cloud_add->points[i].y -= center_y; cloud_add->points[i].z -= center_z;
	}
	//boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_rgb;
	pcl::visualization::PCLVisualizer viewer("Selected matches");
	viewer.addPointCloud(cloud_add, "cloud_view");
	viewer.setBackgroundColor(0, 0, 0);
	float pointsize = 1;
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, pointsize, "cloud_view");
	//
	vector<Vertex> RGB;
	vector<float> temp;
	for (i = 0; i < Hist_match.size(); i++)
		temp.push_back(Hist_match[i].score);
	sort(temp.begin(), temp.end());
	float max_value = temp[temp.size() - 1] * 0.3;
	RGB.resize(Hist_match.size());
	for (i = 0; i < Hist_match.size(); i++)
	{

		float r, g, b;
		float val = Hist_match[i].score * 255 / max_value;
		//green
		if (val < 128)
		{
			g = 0;
		}
		else if (val < 192)
		{
			g = 255.0f / 64 * (val - 128);
		}
		else
		{
			g = 255.0f;
		}
		//blue
		if (val < 64)
		{
			b = 255.0 / 64 * val;
		}
		else if (val < 192)
		{
			b = 255.0;
		}
		else
		{
			b = -255.0 / 63 * (val - 192) + 255;
		}
		//red
		if (val < 64)
		{
			r = 255.0;
		}
		else if (val < 128)
		{
			r = -255.0 / 63 * (val - 64) + 255;
		}
		else
		{
			r = 0.0;
		}
		Vertex temp;
		temp.x = r; temp.y = g; temp.z = b;
		RGB[i] = temp;
	}
	for (int i = 0; i < k; i++)
	{
		std::stringstream ss_line;
		pcl::PointXYZ id1 = Hist_match[i].src;
		pcl::PointXYZ id2 = Hist_match[i].des;
		ss_line << "correspondence_line" << i;
		pcl::PointXYZRGB PointA;
		pcl::PointXYZRGB PointB;
		PointA.x = id1.x - center_x;
		PointA.y = id1.y - center_y;
		PointA.z = id1.z - center_z;
		PointB.x = id2.x - Corres_view_gap * 0.25 * mr - center_x;
		PointB.y = id2.y + Corres_view_gap * 1 * mr - center_y;
		PointB.z = id2.z - center_z;
		double line_R = RGB[i].x;
		double line_G = RGB[i].y;
		double line_B = RGB[i].z;
		double max_channel = std::max(line_R, std::max(line_G, line_B));
		line_R /= max_channel;
		line_G /= max_channel;
		line_B /= max_channel;
		viewer.addLine(PointA, PointB, line_R, line_G, line_B, ss_line.str());
	}
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
        this_thread::sleep_for(100ms);
	}
}
//initial correspondence visualization
void Corres_Viewer_Scorecolor(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_s, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_t, vector<Corre>& Hist_match, float& mr, int k)
{
	int i;
	float gap = Corres_view_gap * mr;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_add(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointXYZRGB tar_temp;
	uint8_t r1(sR), g1 = (sG), b1 = (sB), r2(tR), g2 = (tG), b2 = (tB);
	uint32_t rgb1 = (static_cast<uint32_t>(r1) << 16 |
		static_cast<uint32_t>(g1) << 8 | static_cast<uint32_t>(b1));
	uint32_t rgb2 = (static_cast<uint32_t>(r2) << 16 |
		static_cast<uint32_t>(g2) << 8 | static_cast<uint32_t>(b2));
	for (i = 0; i < cloud_s->points.size(); i++)
	{
		tar_temp.x = cloud_s->points[i].x;
		tar_temp.y = cloud_s->points[i].y;
		tar_temp.z = cloud_s->points[i].z;
		tar_temp.rgb = *reinterpret_cast<float*>(&rgb1);
		cloud_add->points.push_back(tar_temp);
	}
	for (i = 0; i < cloud_t->points.size(); i++)
	{

		tar_temp.x = cloud_t->points[i].x - Corres_view_gap * 1 * mr;
		//tar_temp.y = cloud_t->points[i].y + gap;
		tar_temp.y = cloud_t->points[i].y;//+ Corres_view_gap * 1*mr;
		tar_temp.z = cloud_t->points[i].z;//+ Corres_view_gap * 1 * mr;
		tar_temp.rgb = *reinterpret_cast<float*>(&rgb2);

		cloud_add->points.push_back(tar_temp);
	}
	float center_x = 0, center_y = 0, center_z = 0;
	for (i = 0; i < cloud_add->points.size(); i++)
	{
		center_x += cloud_add->points[i].x; center_y += cloud_add->points[i].y; center_z += cloud_add->points[i].z;
	}
	center_x /= cloud_add->points.size(); center_y /= cloud_add->points.size(); center_z /= cloud_add->points.size();
	for (i = 0; i < cloud_add->points.size(); i++)
	{
		cloud_add->points[i].x -= center_x; cloud_add->points[i].y -= center_y; cloud_add->points[i].z -= center_z;
	}
	//boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_rgb;
	pcl::visualization::PCLVisualizer viewer("match viewer");
	viewer.addPointCloud(cloud_add, "cloud_view");
	viewer.setBackgroundColor(255, 255, 255);
	float pointsize = 1;
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, pointsize, "cloud_view");
	//
	vector<Vertex> RGB;
	vector<float> temp;
	for (i = 0; i < Hist_match.size(); i++)
		temp.push_back(Hist_match[i].score);
	sort(temp.begin(), temp.end());
	float max_value = temp[temp.size() - 1] * 0.3;
	RGB.resize(Hist_match.size());

	srand(time(0));
	for (i = 0; i < Hist_match.size(); i++)
	{
		//float r, g, b;
		//float val = Hist_match[i].score * 255 / max_value;
		////red  
		//if (val<128)
		//{
		//	r = 0;
		//}
		//else if (val<192)
		//{
		//	r = 255.0f / 64 * (val - 128);
		//}
		//else
		//{
		//	r = 255.0f;
		//}
		////green  
		//if (val<64)
		//{
		//	g = 255.0 / 64 * val;
		//}
		//else if (val<192)
		//{
		//	g = 255.0;
		//}
		//else
		//{
		//	g = -255.0 / 63 * (val - 192) + 255;
		//}
		////blue  
		//if (val<64)
		//{
		//	b = 255.0;
		//}
		//else if (val<128)
		//{
		//	b = -255.0 / 63 * (val - 64) + 255;
		//}
		//else
		//{
		//	b = 0.0;
		//}
		float r = rand() % 255;
		float g = rand() % 255;
		float b = rand() % 255;
		Vertex temp;
		temp.x = r; temp.y = g; temp.z = b;
		RGB[i] = temp;
	}
	for (int i = 0; i < k; i++)
	{
		std::stringstream ss_line;
		int id1 = Hist_match[i].source_idx;
		int id2 = Hist_match[i].target_idx;
		ss_line << "correspondence_line" << id1 << "_" << id2;
		pcl::PointXYZRGB PointA = cloud_add->points[id1];
		pcl::PointXYZRGB PointB = cloud_add->points[cloud_s->points.size() + id2];
		double line_R = RGB[i].x;
		double line_G = RGB[i].y;
		double line_B = RGB[i].z;
		double max_channel = std::max(line_R, std::max(line_G, line_B));
		line_R /= max_channel;
		line_G /= max_channel;
		line_B /= max_channel;
		viewer.addLine(PointA, PointB, line_R, line_G, line_B, ss_line.str());
	}
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}
//selected correspondence visualization
void Corres_selected_visual(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_s, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_t, vector<Corre_3DMatch>& Hist_match, float& mr, float GT_thresh, Eigen::Matrix4d& GT_Mat)
{
	int i;
	float gap = Corres_view_gap * mr;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_add(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointXYZRGB tar_temp;
	uint8_t r1(sR), g1 = (sG), b1 = (sB), r2(tR), g2 = (tG), b2 = (tB);
	uint32_t rgb1 = (static_cast<uint32_t>(r1) << 16 |
		static_cast<uint32_t>(g1) << 8 | static_cast<uint32_t>(b1));
	uint32_t rgb2 = (static_cast<uint32_t>(r2) << 16 |
		static_cast<uint32_t>(g2) << 8 | static_cast<uint32_t>(b2));
	for (i = 0; i < cloud_s->points.size(); i++)
	{
		tar_temp.x = cloud_s->points[i].x;
		tar_temp.y = cloud_s->points[i].y;
		tar_temp.z = cloud_s->points[i].z;
		tar_temp.rgb = *reinterpret_cast<float*>(&rgb1);
		cloud_add->points.push_back(tar_temp);
	}
	for (i = 0; i < cloud_t->points.size(); i++)
	{

		tar_temp.x = cloud_t->points[i].x - Corres_view_gap * 0.25 * mr;
		//tar_temp.y = cloud_t->points[i].y + gap;
		tar_temp.y = cloud_t->points[i].y + Corres_view_gap * 1 * mr;
		tar_temp.z = cloud_t->points[i].z;
		tar_temp.rgb = *reinterpret_cast<float*>(&rgb2);

		cloud_add->points.push_back(tar_temp);
	}
	float center_x = 0, center_y = 0, center_z = 0;
	for (i = 0; i < cloud_add->points.size(); i++)
	{
		center_x += cloud_add->points[i].x; center_y += cloud_add->points[i].y; center_z += cloud_add->points[i].z;
	}
	center_x /= cloud_add->points.size(); center_y /= cloud_add->points.size(); center_z /= cloud_add->points.size();
	for (i = 0; i < cloud_add->points.size(); i++)
	{
		cloud_add->points[i].x -= center_x; cloud_add->points[i].y -= center_y; cloud_add->points[i].z -= center_z;
	}
	//boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_rgb;
	pcl::visualization::PCLVisualizer viewer("match viewer");
	viewer.addPointCloud(cloud_add, "cloud_view");
	viewer.setBackgroundColor(255, 255, 255);
	float pointsize = 1;
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, pointsize, "cloud_view");

	//float line_R = line_rgb.x, line_G = line_rgb.y, line_B = line_rgb.z;
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_match(new::pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr tran_source_match(new::pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr target_match(new::pcl::PointCloud<pcl::PointXYZ>);
	double line_R, line_G, line_B;
	for (size_t i = 0; i < Hist_match.size(); i++)
	{
		source_match->points.push_back(Hist_match[i].src);
		target_match->points.push_back(Hist_match[i].des);
	}
	pcl::transformPointCloud(*source_match, *tran_source_match, GT_Mat);

	for (int i = 0; i < Hist_match.size(); i++)
	{
		std::stringstream ss_line;
		ss_line << "correspondence_line" << i;
		pcl::PointXYZ PointA = source_match->points[i];
		PointA.x -= center_x;
		PointA.y -= center_y;
		PointA.z -= center_z;
		if (Distance(tran_source_match->points[i], target_match->points[i]) < GT_thresh)
		{
			line_R = 0;
			line_G = 255;
			line_B = 0;
		}
		else
		{
			//continue;
			line_R = 255;
			line_G = 0;
			line_B = 0;
		}
		/*line_R = 255;
		line_G = 0;
		line_B = 0;*/
		pcl::PointXYZ PointB = target_match->points[i];
		PointB.x -= (center_x + Corres_view_gap * 0.25 * mr);
		PointB.y -= (center_y - Corres_view_gap * 1 * mr);
		PointB.z -= center_z;
		viewer.addLine(PointA, PointB, line_R, line_G, line_B, ss_line.str());
	}
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}
void Corres_initial_visual(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_s, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_t, vector<Corre>& Hist_match, float& mr, Eigen::Matrix4d& GT_Mat)
{
	int i;
	float GT_thresh = 5 * mr;
	float gap = Corres_view_gap * mr;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_add(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointXYZRGB tar_temp;
	uint8_t r1(sR), g1 = (sG), b1 = (sB), r2(tR), g2 = (tG), b2 = (tB);
	uint32_t rgb1 = (static_cast<uint32_t>(r1) << 16 |
		static_cast<uint32_t>(g1) << 8 | static_cast<uint32_t>(b1));
	uint32_t rgb2 = (static_cast<uint32_t>(r2) << 16 |
		static_cast<uint32_t>(g2) << 8 | static_cast<uint32_t>(b2));
	for (i = 0; i < cloud_s->points.size(); i++)
	{
		tar_temp.x = cloud_s->points[i].x;
		tar_temp.y = cloud_s->points[i].y;
		tar_temp.z = cloud_s->points[i].z;
		tar_temp.rgb = *reinterpret_cast<float*>(&rgb1);
		cloud_add->points.push_back(tar_temp);
	}
	for (i = 0; i < cloud_t->points.size(); i++)
	{

		tar_temp.x = cloud_t->points[i].x - Corres_view_gap * 1 * mr;
		//tar_temp.y = cloud_t->points[i].y + gap;
		tar_temp.y = cloud_t->points[i].y;//+ Corres_view_gap * 1*mr;
		tar_temp.z = cloud_t->points[i].z;
		tar_temp.rgb = *reinterpret_cast<float*>(&rgb2);

		cloud_add->points.push_back(tar_temp);
	}
	float center_x = 0, center_y = 0, center_z = 0;
	for (i = 0; i < cloud_add->points.size(); i++)
	{
		center_x += cloud_add->points[i].x; center_y += cloud_add->points[i].y; center_z += cloud_add->points[i].z;
	}
	center_x /= cloud_add->points.size(); center_y /= cloud_add->points.size(); center_z /= cloud_add->points.size();
	for (i = 0; i < cloud_add->points.size(); i++)
	{
		cloud_add->points[i].x -= center_x; cloud_add->points[i].y -= center_y; cloud_add->points[i].z -= center_z;
	}
	//boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_rgb;
	pcl::visualization::PCLVisualizer viewer("match viewer");
	viewer.addPointCloud(cloud_add, "cloud_view");
	viewer.setBackgroundColor(255, 255, 255);
	float pointsize = 1;
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, pointsize, "cloud_view");

	//float line_R = line_rgb.x, line_G = line_rgb.y, line_B = line_rgb.z;
	pcl::PointCloud<pcl::PointXYZ>::Ptr tran_source_match(new::pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*cloud_s, *tran_source_match, GT_Mat);
	double line_R, line_G, line_B;
	for (int i = 0; i < Hist_match.size(); i++)
	{
		std::stringstream ss_line;
		int id1 = Hist_match[i].source_idx;
		int id2 = Hist_match[i].target_idx;
		ss_line << "correspondence_line" << id1 << "_" << id2;
		pcl::PointXYZRGB PointA = cloud_add->points[id1];
		if (Distance(tran_source_match->points[Hist_match[i].source_idx], cloud_t->points[Hist_match[i].target_idx]) < GT_thresh)
		{
			line_R = 0;
			line_G = 255;
			line_B = 0;
		}
		else
		{
			//continue;
			line_R = 255;
			line_G = 0;
			line_B = 0;
		}
		//line_R /= 255;
		//line_G /= 255;
		//line_B /= 255;
		pcl::PointXYZRGB PointB = cloud_add->points[cloud_s->points.size() + id2];
		viewer.addLine(PointA, PointB, line_R, line_G, line_B, ss_line.str());
	}
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}

void visualization(PointCloudPtr &cloud_src, PointCloudPtr &cloud_tar, vector<Corre_3DMatch>&match, Eigen::Matrix4d &Mat, float &resolution)
{
	PointCloudPtr src_kpt(new pcl::PointCloud<pcl::PointXYZ>);
	PointCloudPtr des_kpt(new pcl::PointCloud<pcl::PointXYZ>);
	for (size_t i = 0; i < match.size(); i++)
	{
		src_kpt->push_back(match[i].src);
		des_kpt->push_back(match[i].des);
	}
	pcl::visualization::PCLVisualizer viewer("RANSAC");
	pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_src(cloud_src, 123, 123, 123);
	pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_tar(cloud_tar, 123, 123, 123);
	pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> keyPoint_color_handler_src(src_kpt, 0, 0, 255);
	pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> keyPoint_color_handler_tar(des_kpt, 255, 255, 0);
	pcl::PointCloud<pcl::PointXYZ>::Ptr kpt_trans_src(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*src_kpt, *kpt_trans_src, Mat);
	//Add pointcloud
	viewer.addPointCloud(cloud_src, cloud_color_handler_src, "cloud_src");
	viewer.addPointCloud(cloud_tar, cloud_color_handler_tar, "cloud_tar");
	//Add keyPoint
	viewer.addPointCloud(src_kpt, keyPoint_color_handler_src, "keyPoint_src");
	viewer.addPointCloud(des_kpt, keyPoint_color_handler_tar, "keyPoint_tar");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "keyPoint_src");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "keyPoint_tar");

	int cnt = 0;
	for (size_t j = 0; j < match.size(); ++j)
	{
		pcl::PointXYZ a, b;
		a = match[j].src;
		b = match[j].des;
		double dis = Distance(kpt_trans_src->points[j], b);
		std::stringstream SS_line_b;
		SS_line_b << "line" << j;
		if (dis < 0.1)
		{
			cnt++;
			viewer.addLine< pcl::PointXYZ, pcl::PointXYZ>(a, b, 0, 255, 0, SS_line_b.str());
		}
		else {
			viewer.addLine< pcl::PointXYZ, pcl::PointXYZ>(a, b, 255, 0, 0, SS_line_b.str());
		}
	}

	cout << cnt << "/" << match.size() << endl;
	//viewer.addSphere<pcl::PointXYZ>(cloud_src->points[500], 5 * resolution, "sphere", 0);
	viewer.setBackgroundColor(255, 255, 255);
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_src");

	// Set camera position and orientation
	//viewer.setCameraPosition(-3.68332, 2.94092, 5.71266, 0.289847, 0.921947, -0.256907, 0);
	//viewer.setSize(1280, 1024);
	//transform
	viewer.registerKeyboardCallback(&keyboardEventOccurred, (void*)NULL);
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
		//transform
		if (next_iteration)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trans_src(new pcl::PointCloud<pcl::PointXYZ>);

			viewer.removeAllShapes();
			viewer.removePointCloud("cloud_src");
			viewer.removePointCloud("keyPoint_src");
			pcl::transformPointCloud(*cloud_src, *cloud_trans_src, Mat);

			viewer.addPointCloud(cloud_trans_src, cloud_color_handler_src, "cloud_trans_src");
			viewer.addPointCloud(kpt_trans_src, keyPoint_color_handler_src, "keyPoint_trans_src");
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "keyPoint_trans_src");
			for (size_t j = 0; j < match.size(); ++j)
			{
				pcl::PointXYZ a, b;
				a = kpt_trans_src->points[j];
				b = des_kpt->points[j];
				double dis = Distance(a, b);
				std::stringstream SS_line_b;
				SS_line_b << "line" << j;
				if (dis < 0.1)
				{
					viewer.addLine< pcl::PointXYZ, pcl::PointXYZ>(a, b, 0, 255, 0, SS_line_b.str());
				}
				else {
					viewer.addLine< pcl::PointXYZ, pcl::PointXYZ>(a, b, 255, 0, 0, SS_line_b.str());
				}
			}
		}
		next_iteration = false;
	}
	//system("pause");
}

void visualization(PointCloudPtr& ov_src, PointCloudPtr& cloud_src, PointCloudPtr& cloud_tar, vector<Corre_3DMatch>& match, Eigen::Matrix4d& Mat, Eigen::Matrix4d& GTmat, float& resolution)
{
	PointCloudPtr src_kpt(new pcl::PointCloud<pcl::PointXYZ>);
	PointCloudPtr des_kpt(new pcl::PointCloud<pcl::PointXYZ>);
	for (size_t i = 0; i < match.size(); i++)
	{
		src_kpt->push_back(match[i].src);
		des_kpt->push_back(match[i].des);
	}
	pcl::visualization::PCLVisualizer viewer("RANSAC");
	pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_src(cloud_src, 123, 123, 123);
	pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> cloud_color_handler_tar(cloud_tar, 123, 123, 123);
	pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> keyPoint_color_handler_src(src_kpt, 0, 0, 255);
	pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> keyPoint_color_handler_tar(des_kpt, 255, 255, 0);
	pcl::PointCloud<pcl::PointXYZ>::Ptr kpt_trans_src(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr ov_des(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr ov_trans_src(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*src_kpt, *kpt_trans_src, Mat);
	pcl::transformPointCloud(*ov_src, *ov_des, GTmat);
	pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> ov_color_handler_src(ov_src, 255, 0, 0);
	pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ> ov_color_handler_tar(ov_des, 0, 255, 0);
	//Add pointcloud
	viewer.addPointCloud(cloud_src, cloud_color_handler_src, "cloud_src");
	viewer.addPointCloud(cloud_tar, cloud_color_handler_tar, "cloud_tar");
	//Add keyPoint
	viewer.addPointCloud(src_kpt, keyPoint_color_handler_src, "keyPoint_src");
	viewer.addPointCloud(des_kpt, keyPoint_color_handler_tar, "keyPoint_tar");
	//Add overlap
	viewer.addPointCloud(ov_src, ov_color_handler_src, "ov_src");
	viewer.addPointCloud(ov_des, ov_color_handler_tar, "ov_des");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "keyPoint_src");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "keyPoint_tar");

	int cnt = 0;
	for (size_t j = 0; j < match.size(); ++j)
	{
		pcl::PointXYZ a, b;
		a = match[j].src;
		b = match[j].des;
		double dis = Distance(kpt_trans_src->points[j], b);
		std::stringstream SS_line_b;
		SS_line_b << "line" << j;
		if (dis < 0.1)
		{
			cnt++;
			viewer.addLine< pcl::PointXYZ, pcl::PointXYZ>(a, b, 0, 255, 0, SS_line_b.str());
		}
		else {
			viewer.addLine< pcl::PointXYZ, pcl::PointXYZ>(a, b, 255, 0, 0, SS_line_b.str());
		}
	}

	cout << cnt << "/" << match.size() << endl;
	//viewer.addSphere<pcl::PointXYZ>(cloud_src->points[500], 5 * resolution, "sphere", 0);
	viewer.setBackgroundColor(255, 255, 255);
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_src");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "ov_src");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "ov_des");

	// Set camera position and orientation
	//viewer.setCameraPosition(-3.68332, 2.94092, 5.71266, 0.289847, 0.921947, -0.256907, 0);
	//viewer.setSize(1280, 1024);
	//transform
	viewer.registerKeyboardCallback(&keyboardEventOccurred, (void*)NULL);
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
		//transform
		if (next_iteration)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trans_src(new pcl::PointCloud<pcl::PointXYZ>);

			viewer.removeAllShapes();
			viewer.removePointCloud("cloud_src");
			viewer.removePointCloud("keyPoint_src");
			viewer.removePointCloud("ov_src");
			pcl::transformPointCloud(*cloud_src, *cloud_trans_src, Mat);
			pcl::transformPointCloud(*ov_src, *ov_trans_src, Mat);

			viewer.addPointCloud(cloud_trans_src, cloud_color_handler_src, "cloud_trans_src");
			viewer.addPointCloud(ov_trans_src, ov_color_handler_src, "ov_trans_src");
			viewer.addPointCloud(kpt_trans_src, keyPoint_color_handler_src, "keyPoint_trans_src");
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "keyPoint_trans_src");
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "ov_trans_src");
			for (size_t j = 0; j < match.size(); ++j)
			{
				pcl::PointXYZ a, b;
				a = kpt_trans_src->points[j];
				b = des_kpt->points[j];
				double dis = Distance(a, b);
				std::stringstream SS_line_b;
				SS_line_b << "line" << j;
				if (dis < 0.1)
				{
					viewer.addLine< pcl::PointXYZ, pcl::PointXYZ>(a, b, 0, 255, 0, SS_line_b.str());
				}
				else {
					viewer.addLine< pcl::PointXYZ, pcl::PointXYZ>(a, b, 255, 0, 0, SS_line_b.str());
				}
			}
		}
		next_iteration = false;
	}
	//system("pause");
}