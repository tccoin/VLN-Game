# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# examples/python/t_reconstruction_system/dense_slam_gui.py

# P.S. This example is used in documentation, so, please ensure the changes are
# synchronized.

import open3d as o3d
import open3d.core as o3c
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from arguments import get_args
import os
import numpy as np
import threading
import time
import torch
from multiprocessing import Process, Queue, set_start_method
import cv2
from PIL import Image

from utils.compute_similarities import color_by_clip_sim
from utils.slam_classes import DetectionList, MapObjectList

def set_enabled(widget, enable):
    widget.enabled = enable
    for child in widget.get_children():
        child.enabled = enable

class ReconstructionWindow:

    def __init__(self, args, font_id):
        self.args = args

        self.device = "cuda:0"

        self.window = gui.Application.instance.create_window(
            'Open3D - Reconstruction', 1280, 800)

        w = self.window
        em = w.theme.font_size

        spacing = int(np.round(0.25 * em))
        vspacing = int(np.round(0.5 * em))

        margins = gui.Margins(vspacing)

        # First panel
        self.panel = gui.Vert(spacing, margins)

        ## Items in adjustable props
        self.adjustable_prop_grid = gui.VGrid(2, spacing,
                                              gui.Margins(em, 0, em, 0))

        ### Update surface?
        rgb_pc_label = gui.Label('RGB PC?')
        self.rgb_pc_box = gui.Checkbox('')
        self.rgb_pc_box.checked = True
        self.adjustable_prop_grid.add_child(rgb_pc_label)
        self.adjustable_prop_grid.add_child(self.rgb_pc_box)

        ### Show trajectory?
        trajectory_label = gui.Label('Trajectory?')
        self.trajectory_box = gui.Checkbox('')
        self.trajectory_box.checked = True
        self.adjustable_prop_grid.add_child(trajectory_label)
        self.adjustable_prop_grid.add_child(self.trajectory_box)

        ## Application control
        b = gui.ToggleSwitch('Resume/Pause')
        b.set_on_clicked(self._on_switch)

        self.text_edit = gui.TextEdit()
        self.submit_button = gui.Button("Submit")
        self.submit_button.set_on_clicked(self._on_submit)
        self.input_text = None

        ## Tabs
        tab_margins = gui.Margins(0, int(np.round(0.5 * em)), 0, 0)
        tabs = gui.TabControl()


        ### Rendered image tab
        tab2 = gui.Vert(0, tab_margins)
        self.semantic_image = gui.ImageWidget()
        self.map_image = gui.ImageWidget()
        tab2.add_child(self.semantic_image)
        tab2.add_fixed(vspacing)
        tab2.add_child(self.map_image)
        tabs.add_tab('Semantic images', tab2)

        ### Input image tab
        tab1 = gui.Vert(0, tab_margins)
        self.input_color_image = gui.ImageWidget()
        self.input_depth_image = gui.ImageWidget()
        tab1.add_child(self.input_color_image)
        tab1.add_fixed(vspacing)
        tab1.add_child(self.input_depth_image)
        tabs.add_tab('Input images', tab1)

        ### Info tab
        tab3 = gui.Vert(0, tab_margins)
        self.output_info = gui.Label('Output info')
        self.output_info.font_id = font_id
        tab3.add_child(self.output_info)
        tabs.add_tab('Info', tab3)

        self.panel.add_child(gui.Label('Reconstruction settings'))
        self.panel.add_child(self.adjustable_prop_grid)
        self.panel.add_child(self.text_edit)
        self.panel.add_child(self.submit_button)
        self.panel.add_child(b)
        self.panel.add_stretch()
        self.panel.add_child(tabs)


        # Scene widget
        self.widget3d = gui.SceneWidget()

        # FPS panel
        self.fps_panel = gui.Vert(spacing, margins)
        self.output_fps = gui.Label('FPS: 0.0')
        self.fps_panel.add_child(self.output_fps)

        # Now add all the complex panels
        w.add_child(self.panel)
        w.add_child(self.widget3d)
        w.add_child(self.fps_panel)

        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene.set_background([1, 1, 1, 1])

        w.set_on_layout(self._on_layout)
        w.set_on_close(self._on_close)

        self.saved_objects = None
        self.saved_full_points = None
        self.saved_full_colors = None

        self.is_done = False

        self.is_started = False
        self.is_running = True

        self.idx = 0
        self.traj = []

        threading.Thread(name='UpdateMain', target=self.update_main).start()

    def _on_submit(self):
        input_text = self.text_edit.text_value
        print("Input text:", input_text)
        self.send_queue.put([input_text, self.is_running])

    def _on_layout(self, ctx):
        em = ctx.theme.font_size

        panel_width = 20 * em
        rect = self.window.content_rect

        self.panel.frame = gui.Rect(rect.x, rect.y, panel_width, rect.height)

        x = self.panel.frame.get_right()
        self.widget3d.frame = gui.Rect(x, rect.y,
                                       rect.get_right() - x, rect.height)

        fps_panel_width = 7 * em
        fps_panel_height = 2 * em
        self.fps_panel.frame = gui.Rect(rect.get_right() - fps_panel_width,
                                        rect.y, fps_panel_width,
                                        fps_panel_height)

    # Toggle callback: application's main controller
    def _on_switch(self, is_on):
        
        self.is_running = not self.is_running
        self.send_queue.put([None, self.is_running])


    def _on_close(self):
        

        return True
    
    def plane_segmentation(self, pcd, distance_threshold=0.05, ransac_n=3, num_iterations=50):
        # 使用RANSAC拟合平面
        ground_sum = []
        while True:
            plane_model, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)
            if len(inliers) < 100:
                break
            inlier_cloud = pcd.select_by_index(inliers)
            outlier_cloud = pcd.select_by_index(inliers, invert=True)
            pcd = pcd.select_by_index(inliers, invert=True)
            # pcd.paint_uniform_color([0, 1, 0])  # 绿色表示地面点
            ground_sum.append(inlier_cloud)
            
        
        # ground_cloud = o3d.t.geometry.PointCloud(o3c.Tensor(np.vstack(ground_sum)))
        return ground_sum


    def init_render(self):
       
        self.window.set_needs_layout()

        bbox = o3d.geometry.AxisAlignedBoundingBox([-5, -5, -5], [5, 5, 5])
        self.widget3d.setup_camera(90, bbox, [0, 0, 0])
        # self.widget3d.setup_camera(90, bbox, [camera_matrix[3,0], camera_matrix[3,1], camera_matrix[3,2]])[0, 0, 0]
        # self.widget3d.look_at(camera_matrix[:3,0], camera_matrix[:3,1], camera_matrix[:3,2])
        self.widget3d.look_at([0, 0, 0], [0, -1, -3], [0, -1, 0])

        points = np.random.rand(100, 3)
        colors = np.zeros((100, 3))
        colors[:, 0] = 1  # 红色
        pcd_t = o3d.t.geometry.PointCloud(
                    o3c.Tensor(points.astype(np.float32)))
        pcd_t.point.colors = o3c.Tensor(colors)
        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        self.widget3d.scene.add_geometry('points', pcd_t, material)  # Add material argument

        # Add a coordinate frame
        self.widget3d.scene.show_axes(True)

    def update_render(self, 
                      objects, 
                      point_sum_points,
                      point_sum_colors):
        
        self.window.set_needs_layout()
        bbox = o3d.geometry.AxisAlignedBoundingBox([-5, -5, -5], [5, 5, 5])
        self.widget3d.setup_camera(90, bbox, [0, 0, 0])
        self.widget3d.look_at([0, 0, 0], [-3, 3, 0], [0, 1, 0])

        self.widget3d.scene.show_axes(True)


        if len(objects) > 0:
            pcd_sum = []
            pcd_color_sum = []
            bbox_sum = []
            bbox_color_sum = []
            
            # clip_objects = MapObjectList(device=self.device)
            # clip_objects.load_serializable(objects)
            # text_queries = "stair"
            # clip_objects, similarities = color_by_clip_sim(text_queries, 
            #                                             clip_objects, 
            #                                             self.clip_model, 
            #                                             self.clip_tokenizer)
            
            for obj in objects:
                pcd_sum.append(obj['pcd_np'].astype(np.float32))
                pcd_color_sum.append(obj['pcd_color_np'].astype(np.float32))
                bbox_sum.append((obj['bbox_np'][0].astype(np.float32), obj['bbox_np'][1].astype(np.float32)))
            # print("bbox_sum: ", bbox_sum)
            pcd = o3d.t.geometry.PointCloud(
                o3c.Tensor(np.vstack(pcd_sum)))
            pcd.point.colors = o3c.Tensor(np.vstack(pcd_color_sum))
            

            self.widget3d.scene.remove_geometry("points")
            material = rendering.MaterialRecord()
            material.shader = "defaultUnlit"
            self.widget3d.scene.add_geometry('points', pcd, material)  # Add material argument


            # bboxs = [o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound) for min_bound, max_bound in bbox_sum]
            # mat = rendering.MaterialRecord()
            # mat.shader = "unlitLine"
            # for i, bbox in enumerate(bboxs):
            #     bbox.color = [1, 0, 0]
            #     self.widget3d.scene.remove_geometry(f"bbox_{i}")
            #     self.widget3d.scene.add_geometry(f"bbox_{i}", bbox, mat)


        self.widget3d.scene.remove_geometry("full_pcd")
        if self.rgb_pc_box.checked:
            full_pcd = o3d.t.geometry.PointCloud(
            o3c.Tensor(point_sum_points.astype(np.float32)))
            full_pcd.point.colors = o3c.Tensor(point_sum_colors.astype(np.float32))
            
            material = rendering.MaterialRecord()
            material.shader = "defaultUnlit"
            self.widget3d.scene.add_geometry('full_pcd', full_pcd, material)  # Add material argument
            
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(point_sum_points.astype(np.float32))
            # hull, _ = pcd.compute_convex_hull()
            # hull_lines = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
            # hull_lines.paint_uniform_color([1, 0, 0]) 
            # mat = rendering.MaterialRecord()
            # mat.shader = "unlitLine"
            # self.widget3d.scene.add_geometry(f"hull", hull_lines, mat)
            
        # self.widget3d.scene.remove_geometry("ground_cloud")
        # # self.widget3d.scene.remove_geometry("non_ground_cloud")
        # if self.rgb_pc_box.checked:
        #     full_pcd = o3d.t.geometry.PointCloud(
        #     o3c.Tensor(point_sum_points.astype(np.float32)))
        #     full_pcd.point.colors = o3c.Tensor(point_sum_colors.astype(np.float32))
        #     # full_pcd.voxel_down_sample(10)
        #     ground_time = time.time()

        #     ground_cloud = self.plane_segmentation(full_pcd)
                
        #     ground_f_time = time.time()
        #     print('ground_time: %.3f秒'%(ground_f_time - ground_time)) 
        #     # non_ground_cloud.paint_uniform_color([1, 0, 0])

        #     material = rendering.MaterialRecord()
        #     material.shader = "defaultUnlit"
        #     self.widget3d.scene.add_geometry('ground_cloud', ground_cloud, material)  # Add material argument
        #     # self.widget3d.scene.add_geometry('non_ground_cloud', non_ground_cloud, material)  # Add material argument


        # 保存图片
        # self.widget3d.scene.scene.render_to_image(self.on_image_rendered)

        # # 保存视频
        # video_filename = "scene_widget_video.mp4"
        # recorder = rendering.VideoRecorder(video_filename, 30, 1024, 768)
            
     





    # Major loop
    def update_main(self):
        
        data = np.load(self.args.path_npz + 'arrays.npz', allow_pickle=True)

        objects = data['objects']
        point_sum_points = data['full_points']
        point_sum_colors = data['full_colors']

        gui.Application.instance.post_to_main_thread(
            self.window, lambda: self.update_render(
                objects, 
                point_sum_points,
                point_sum_colors)
                )

  


if __name__ == '__main__':
    args = get_args()

    app = gui.Application.instance
    app.initialize()
    mono = app.add_font(gui.FontDescription(gui.FontDescription.MONOSPACE))
    w = ReconstructionWindow(args, mono)
    app.run()
