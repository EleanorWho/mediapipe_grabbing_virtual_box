import moderngl
import moderngl_window as mglw
from pyrr import Matrix44

import cv2
import numpy as np
import os
from array import array

from prediction import predict, get_camera_matrix, get_fov_y, solvepnp
import time


class CameraAR(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "CameraAR"
    resource_dir = os.path.normpath(os.path.join(__file__, '../data'))
    previousTime = 0
    currentTime = 0
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Shader for rendering 3D objects
        self.prog3d = self.ctx.program(
            vertex_shader='''
                #version 330

                uniform mat4 Mvp;

                in vec3 in_position;
                in vec3 in_normal;
                in vec2 in_texcoord_0;

                out vec3 v_vert;
                out vec3 v_norm;
                out vec2 v_text;

                void main() {
                    gl_Position = Mvp * vec4(in_position, 1.0);
                    v_vert = in_position;
                    v_norm = in_normal;
                    v_text = in_texcoord_0;
                }
            ''',
            fragment_shader='''
                #version 330

                uniform vec3 Color;
                uniform vec3 Light;
                uniform sampler2D Texture;
                uniform bool withTexture;

                in vec3 v_vert;
                in vec3 v_norm;
                in vec2 v_text;

                out vec4 f_color;

                void main() {
                    float lum = clamp(dot(normalize(Light - v_vert), normalize(v_norm)), 0.0, 1.0) * 0.8 + 0.2;
                    if (withTexture) {
                        f_color = vec4(Color * texture(Texture, v_text).rgb * lum, 1.0);
                    } else {
                        f_color = vec4(Color * lum, 1.0);
                    }
                }
            ''',
        )
        self.mvp = self.prog3d['Mvp']
        self.light = self.prog3d['Light']
        self.color = self.prog3d['Color']
        self.withTexture = self.prog3d['withTexture']

        # Load the 3D virtual object, and the marker for hand landmarks
        self.scene_cube = self.load_scene('crate.obj')
        self.scene_marker = self.load_scene('marker.obj')

        # Extract the VAOs from the scene
        self.vao_cube = self.scene_cube.root_nodes[0].mesh.vao.instance(self.prog3d)
        self.vao_marker = self.scene_marker.root_nodes[0].mesh.vao.instance(self.prog3d)

        # Texture of the cube
        self.texture = self.load_texture_2d('crate.png')
        
        # Define the initial position of the virtual object
        # The OpenGL camera is position at the origin, and look at the negative Z axis. The object is at 30 centimeters in front of the camera. 
        self.object_pos = np.array([0.0, 0.0, -30.0])

        # shader for markers
        self.marker_shader = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 inPosition;
                uniform mat4 orthoMatrix;
                void main() {
                    gl_Position = orthoMatrix * vec4(inPosition, 0.0, 1.0);
                    gl_PointSize = 50.0;
                }
            ''',
            fragment_shader='''
                #version 330
                uniform vec3 Color;
                out vec4 f_color;
                void main() {
                    f_color = vec4(Color, 1.0);
                }
            '''
        )
        quad_vertices = np.array([
            -2, -2,
            2, -2,
            -2,  2,
            2,  2,
        ], dtype='f4')
        # bind vbo and vao of the 2D markers
        self.vbo_2d = self.ctx.buffer(quad_vertices.tobytes())

        self.vao_2d = self.ctx.vertex_array(self.marker_shader, [
            (self.vbo_2d, '2f', 'inPosition')
        ])
        
        """
        --------------------------------------------------------------------
        TODO: Task 3. 
        Add support to render a rectangle of window size. 
        --------------------------------------------------------------------
        """

        # create video shader to render the window with video frames
        self.video_shader = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_vert;
                in vec2 in_text;
                out vec2 v_text;
                void main() {
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                    v_text = in_text;
                }
            ''',
            fragment_shader='''
                #version 330
                uniform sampler2D Texture;
                in vec2 v_text;
                out vec4 f_color;
                void main() {
                    f_color = texture(Texture, v_text);
                }
            '''
        )

        self.video_shader['Texture'].value = 0  # Bind texture to texture unit 0

        # Define the rectangle covering the entire screen
        self.quad_vbo = self.ctx.buffer(np.array([
            -1.0, -1.0,  0.0, 1.0,
            1.0, -1.0,  1.0, 1.0,
            1.0,  1.0,  1.0, 0.0,
            -1.0,  1.0,  0.0, 0.0,
        ], dtype='f4').tobytes())

        self.quad_vao = self.ctx.vertex_array(self.video_shader, [
            (self.quad_vbo, '2f 2f', 'in_vert', 'in_text'),
        ])

        # Start OpenCV camera 
        self.capture = cv2.VideoCapture(0)
        
        # Get a frame to set the window size and aspect ratio
        ret, frame = self.capture.read() 
        self.aspect_ratio = float(frame.shape[1]) / frame.shape[0]
        self.window_size = (int(720.0 * self.aspect_ratio), 720)

        # Create a texture for the video frame. Initialize it with a dummy frame for now.
        self.frame_texture = self.ctx.texture(self.window_size, 3, None)

    def render(self, time: float, frame_time: float):
        self.ctx.clear(1.0, 1.0, 1.0)
        # self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.viewport = (0, 0, self.window_size[0], self.window_size[1])

        """
        ---------------------------------------------------------------
        TODO: Task 3. 
        Get OpenCV video frame, display in OpenGL. 
        Render the frame to a screen-sized rectange. 
        ---------------------------------------------------------------
        """
        # Capture a frame
        ret, frame = self.capture.read()
        if not ret:
            return  # Skip the frame if capture failed

        # Convert frame to RGB (OpenCV uses BGR by default)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)

        """
        ---------------------------------------------------------------
        TODO: Task 4.
        Perform hand landmark prediction, and 
        solve PnP to get world landmarks list.
        ---------------------------------------------------------------
        """

        landmarks = predict(frame)

        # Solve the landmarks in world space
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        camera_matrix = get_camera_matrix(frame_width, frame_height)

        # if landmarks.hand_landmarks:
        #     print("detected")
        # else:
        #     print("nothing detected")

        world_landmarks_list = solvepnp(landmarks.hand_world_landmarks, landmarks.hand_landmarks, camera_matrix, frame_width, frame_height)

        # OpenCV to OpenGL conversion
        # The world points from OpenCV need some changes to be OpenGL ready. 
        # First, the model points are in meters (MediaPipe convention), while our camera matrix is in units. There exists a scale ambiguity of the true hand landmarks, i.e., if we scale up the world points by 1000, its projection remains the same (due to perspective division). 
        # Here we shift the measurement from meter to centimeter, and assume our world space in OpenGL is in centimeters, just for easy visualization and object interaction. So we multiply all points by 100.
        
        # Second, the OpenCV and OpenGL camera coordinate system are different. # OpenCV: right x, down y, into screen z. Image: right x, down y.  
        # OpenGL: right x, up y, out of screen z. Image: right x, up y.
        # Check for image and 3D points flip to make sure the points are properly converted. 
        
        """
        ----------------------------------------------------------------------
        TODO: Task 5.
        We detect a simple pinch gesture, and check if the index finger hits 
        the cube. We approximate by just checking the finger tip is close 
        enough to the cube location.
        ----------------------------------------------------------------------
        """
        grabbed = False
        # It is recommended to work on this task last after all landmarks are in place.

        if len(world_landmarks_list) > 0 and world_landmarks_list[0].shape[0] > 4:
            # Check for pinch gesture
            thumb_tip = world_landmarks_list[0][4]  # index 4 is thumb tip
            index_tip = world_landmarks_list[0][8]  # index 8 is index finger tip

            # Calculate the distance between thumb tip and index finger tip
            pinch_distance = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))
            print(pinch_distance)

            # Define a threshold distance for pinching
            pinch_threshold = 0.03
            scale_factor = 50       # the scale factor is decided after several test

            if pinch_distance < pinch_threshold:
                # Pinch detected
                grabbed = True
                print("grabbed")
                # move the cube to where the index finger is
                self.object_pos = np.array([index_tip[0] * scale_factor, -index_tip[1] * scale_factor, -index_tip[2] * scale_factor])
            else:
                grabbed = False
        
        """
        ----------------------------------------------------------------------
        TODO: Task 4. 
        Render the markers.
        ----------------------------------------------------------------------
        """

        # Note we have to set the OpenGL projection matrix by following parameters from the OpenCV camera matrix, i.e., the field of view.
        # You can use Matrix44.perspective_projection function, and set the parameters accordingly. Note that the fov must be computed based on the camera matrix. See prediction.py. 
        
        # In this example, a random FOV value is set. Do not use this value in your final program. 
        camera_matrix = get_camera_matrix(frame.shape[1], frame.shape[0])
        fov_y = get_fov_y(camera_matrix, frame.shape[0])
        proj = Matrix44.perspective_projection(fov_y, self.aspect_ratio, 0.1, 1000)
        
        # Translate the object to its position 
        translate = Matrix44.from_translation(self.object_pos)
        
        # Add a bit of random rotation just to be dynamic
        rotate = Matrix44.from_y_rotation(np.sin(time) * 0.5 + 0.2)
        
        # Scale the object up for easy viewing
        scale = Matrix44.from_scale((3, 3, 3))
        
        mvp = proj * translate * rotate * scale
        self.color.value = (1.0, 1.0, 1.0)
        if grabbed: # A bit of feedback when the object is grabbed
            self.color.value = (1.0, 0.0, 0.0)
        self.light.value = (10, 10, 10)
        self.mvp.write(mvp.astype('f4'))
        self.withTexture.value = True

        # Update the texture with the new video frame
        frame = cv2.resize(frame, self.window_size)
        self.frame_texture.write(frame.tobytes())

        # disable depth_test to let the camera frame shown at the bottom of the scene
        self.ctx.disable(moderngl.DEPTH_TEST)
        # Render the video frame rectangle
        self.frame_texture.use(location=0)
        self.quad_vao.render(moderngl.TRIANGLE_FAN)

        # Render the landmarks
        # 2D markers
        for landmark in landmarks.hand_landmarks:
            for i in range(len(landmark)):
                gl_x = landmark[i].x * self.window_size[0]
                gl_y = self.window_size[1] - landmark[i].y * self.window_size[1]
                # print(gl_x, gl_y)
                translation = Matrix44.from_translation([gl_x, gl_y, 0.0])

                self.marker_shader['Color'].value = (1.0, 0.0, 0.0)
                ortho_matrix = Matrix44.orthogonal_projection(
                    0, self.window_size[0], 0, self.window_size[1], -1, 1
                )
                self.marker_shader['orthoMatrix'].write((ortho_matrix * translation).astype('f4'))
                self.vao_2d.render(moderngl.TRIANGLE_FAN)

        # enable depth_test to show all the other objects
        self.ctx.enable(moderngl.DEPTH_TEST)
        # Render the object
        self.texture.use()
        self.vao_cube.render()

        # render the markers
        # 3D markers
        for landmark in world_landmarks_list:
            for i in range(len(landmark)):
                gl_x, gl_y, gl_z = landmark[i]
                # print(gl_x, gl_y, gl_z)

                scale_factor = 9
                model_matrix = Matrix44.from_translation([gl_x * scale_factor, -gl_y * scale_factor, -gl_z * scale_factor])
                model_matrix = model_matrix * Matrix44.from_scale((0.02, 0.02, 0.02))
                mvp_matrix = proj * model_matrix
                self.mvp.write(mvp_matrix.astype('f4'))

                self.prog3d['Color'].value = (0.0, 1.0, 0.0)
                self.vao_marker.render()

if __name__ == '__main__':
    CameraAR.run()
