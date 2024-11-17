package com.ece420.lab7;


import android.app.Activity;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.Manifest;
import android.os.Bundle;
import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.SeekBar;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;

import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfInt;
import org.opencv.core.Point;
import org.opencv.core.Rect2d;

import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.tracking.TrackerKCF;

import java.lang.Math;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.ArrayList;
import java.util.stream.IntStream;

public class MainActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "MainActivity";

    // UI Variables
    private Button controlButton;
    private SeekBar colorSeekbar;
    private SeekBar widthSeekbar;
    private SeekBar heightSeekbar;
    private TextView widthTextview;
    private TextView heightTextview;

    // Declare OpenCV based camera view base
    private CameraBridgeViewBase mOpenCvCameraView;
    // Camera size
    private int myWidth;
    private int myHeight;

    // Mat to store RGBA and Grayscale camera preview frame
    private Mat mRgba;
    private Mat mGray;

    // KCF Tracker variables
    private TrackerKCF myTracker;
    private Rect2d myROI = new Rect2d(0,0,0,0);

    private int myROIWidth = 70;
    private int myROIHeight = 70;
    private Scalar myROIColor = new Scalar(0,0,0);
    private int tracking_flag = -1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);
        super.setRequestedOrientation (ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);

        // Request User Permission on Camera
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED){
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA}, 1);}

        // OpenCV Loader and Avoid using OpenCV Manager
        if (!OpenCVLoader.initDebug()) {
            Log.e(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), not working.");
        } else {
            Log.d(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), working.");
        }

        // Setup color seek bar
        colorSeekbar = (SeekBar) findViewById(R.id.colorSeekBar);
        colorSeekbar.setProgress(50);
        setColor(50);
        colorSeekbar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener()
        {
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser)
            {
                setColor(progress);
            }
            public void onStartTrackingTouch(SeekBar seekBar) {}
            public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        // Setup width seek bar
        widthTextview = (TextView) findViewById(R.id.widthTextView);
        widthSeekbar = (SeekBar) findViewById(R.id.widthSeekBar);
        widthSeekbar.setProgress(myROIWidth - 20);
        widthSeekbar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener()
        {
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser)
            {
                // Only allow modification when not tracking
                if(tracking_flag == -1) {
                    myROIWidth = progress + 20;
                }
            }
            public void onStartTrackingTouch(SeekBar seekBar) {}
            public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        // Setup width seek bar
        heightTextview = (TextView) findViewById(R.id.heightTextView);
        heightSeekbar = (SeekBar) findViewById(R.id.heightSeekBar);
        heightSeekbar.setProgress(myROIHeight - 20);
        heightSeekbar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener()
        {
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser)
            {
                // Only allow modification when not tracking
                if(tracking_flag == -1) {
                    myROIHeight = progress + 20;
                }
            }
            public void onStartTrackingTouch(SeekBar seekBar) {}
            public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        // Setup control button
        controlButton = (Button)findViewById((R.id.controlButton));
        controlButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (tracking_flag == -1) {
                    // Modify UI
                    controlButton.setText("STOP");
                    widthTextview.setVisibility(View.INVISIBLE);
                    widthSeekbar.setVisibility(View.INVISIBLE);
                    heightTextview.setVisibility(View.INVISIBLE);
                    heightSeekbar.setVisibility(View.INVISIBLE);
                    // Modify tracking flag
                    tracking_flag = 0;
                }
                else if(tracking_flag == 1){
                    // Modify UI
                    controlButton.setText("START");
                    widthTextview.setVisibility(View.VISIBLE);
                    widthSeekbar.setVisibility(View.VISIBLE);
                    heightTextview.setVisibility(View.VISIBLE);
                    heightSeekbar.setVisibility(View.VISIBLE);
                    // Tear down myTracker
                    myTracker.clear();
                    // Modify tracking flag
                    tracking_flag = -1;
                }
            }
        });

        // Setup OpenCV Camera View
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.opencv_camera_preview);
        // Use main camera with 0 or front camera with 1
        mOpenCvCameraView.setCameraIndex(0);

        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);

        // Force camera resolution, ignored since OpenCV automatically select best ones
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.enableView();

        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }


    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    // Helper Function to map single integer to color scalar
    // https://www.particleincell.com/2014/colormap/
    public void setColor(int value) {
        double a=(1-(double)value/100)/0.2;
        int X=(int)Math.floor(a);
        int Y=(int)Math.floor(255*(a-X));
        double newColor[] = {0,0,0};
        switch(X)
        {
            case 0:
                // r=255;g=Y;b=0;
                newColor[0] = 255;
                newColor[1] = Y;
                break;
            case 1:
                // r=255-Y;g=255;b=0
                newColor[0] = 255-Y;
                newColor[1] = 255;
                break;
            case 2:
                // r=0;g=255;b=Y
                newColor[1] = 255;
                newColor[2] = Y;
                break;
            case 3:
                // r=0;g=255-Y;b=255
                newColor[1] = 255-Y;
                newColor[2] = 255;
                break;
            case 4:
                // r=Y;g=0;b=255
                newColor[0] = Y;
                newColor[2] = 255;
                break;
            case 5:
                // r=255;g=0;b=255
                newColor[0] = 255;
                newColor[2] = 255;
                break;
        }
        myROIColor.set(newColor);
        return;
    }

    // OpenCV Camera Functionality Code
    @Override
    public void onCameraViewStarted(int width, int height) {
        Log.d(TAG, "onCameraViewStarted");
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);
        myWidth = width;
        myHeight = height;
        myROI = new Rect2d(myWidth / 2 - myROIWidth / 2,
                            myHeight / 2 - myROIHeight / 2,
                            myROIWidth,
                            myROIHeight);
    }

    @Override
    public void onCameraViewStopped() {
        Log.d(TAG, "onCameraViewStopped");
        mRgba.release();
        mGray.release();
    }

    private double dist_point_line(double x1, double y1, double x2, double y2, double x0, double y0){
        return ((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / (double) Math.sqrt(Math.pow(y2-y1, 2) + Math.pow(x2-x1, 2));
    }

    private void quadrangle_approx(MatOfPoint orig_contour, MatOfPoint quadrangle) {
        Point[] points = orig_contour.toArray();
        int num_pts = points.length;

        if(num_pts < 4) {
            return;
        }

        // Get Manhattan distance between adjacent points
        ArrayList<Integer> distances = new ArrayList<>(num_pts);

        distances.add(Math.abs((int) points[0].x - (int) points[num_pts-1].x) + Math.abs((int) points[0].y - (int) points[num_pts-1].y));
        for (int i = 1; i < num_pts; i++) {
            int dist = Math.abs((int) points[i].x - (int) points[i-1].x) + Math.abs((int) points[i].y - (int) points[i-1].y);
            distances.add(dist);
        }

        double lines[][] = new double[4][4];   // Could be int
        double corners[][] = new double[4][2]; // Could be int
        double slope_array[] = new double[4];  // Could be float
        double y_int_array[] = new double[4];  // Could be float

        // Sort indices based on distance
//        ArrayList<Integer> dist_idx = new ArrayList<>(num_pts);
//        dist_idx.set(0, 0); // TODO argsort
//
//        for(int i = 0; i < num_pts; i++) {
//            dist_idx.set(i, i);
//        }

        int dist_idx[] = IntStream.range(0, num_pts)
                .boxed().sorted((i, j) -> Integer.compare(distances.get(i), distances.get(j)))
                .mapToInt(ele -> ele).toArray();

        // Find 4 longest, non-colinear lines in contours (will segments on card edges)
        int COLINEAR_CUTOFF = 10;
        int next_line = 0;
        for(int i = 0; i < num_pts; i++) {
            double slope;
            double y_int;
            int max_idx = dist_idx[num_pts - (i+1)];

            lines[next_line][0] = points[max_idx].x;
            lines[next_line][1] = points[max_idx].y;
            if(max_idx == 0) {
                lines[next_line][2] = points[num_pts - 1].x;
                lines[next_line][3] = points[num_pts - 1].y;
            } else {
                lines[next_line][2] = points[max_idx - 1].x;
                lines[next_line][3] = points[max_idx - 1].y;
            }

            double slope_denom = (lines[next_line][2]-lines[next_line][0]);
            if(slope_denom == 0.0) {
                slope_denom = 0.001;
            }

            slope = (lines[next_line][3]-lines[next_line][1]) / slope_denom;
            y_int = lines[next_line][1] - slope * lines[next_line][0];

            boolean colinear = false;
            for(int j = 0; j < next_line; j++) {
                if(dist_point_line(lines[j][0], lines[j][1], lines[j][2], lines[j][3], lines[next_line][0], lines[next_line][1]) < COLINEAR_CUTOFF &&
                        dist_point_line(lines[j][0], lines[j][1], lines[j][2], lines[j][3], lines[next_line][2], lines[next_line][3]) < COLINEAR_CUTOFF) {
                    colinear = true;
                    break;
                }
            }

            if(!colinear) {
                slope_array[next_line] = slope;
                y_int_array[next_line] = y_int;

                next_line++;
                if(next_line == 4) {
                    break;
                }
            }
        }

        // Find opposite opposite to lines[0]
        if(Math.abs(slope_array[0]) < 0.001) {
            slope_array[0] = 0.001;
        }

        int opposite_idx = 1;
        double slope_ratio = Math.abs(Math.abs(slope_array[1] / slope_array[0]) - 1);
        for(int i = 1; i < 4; ++i) {
            double new_slope_ratio = Math.abs(Math.abs(slope_array[i] / slope_array[0]) - 1);

            if(new_slope_ratio < slope_ratio){
                slope_ratio = new_slope_ratio;
                opposite_idx = i;
            }
        }

        // Find 2 line indicies perpendicular to lines[0]
        int adj_idx[];
        if(opposite_idx == 1) {
            adj_idx = new int[]{2, 3};
        } else if(opposite_idx == 2) {
            adj_idx = new int[]{1, 3};
        } else {
            adj_idx = new int[]{1,2};
        }

        for(int i = 0; i < 2; i++) {
            double slope_denom;
            double x_tmp;

            slope_denom = (slope_array[0] - slope_array[adj_idx[i]]);
            if(slope_denom == 0) {
                slope_denom = 0.001;
            }

            x_tmp = (y_int_array[adj_idx[i]] - y_int_array[0]) / slope_denom;
            corners[i][0] = x_tmp;
            corners[i][1] = slope_array[0] * x_tmp + y_int_array[0];

            slope_denom = (slope_array[opposite_idx] - slope_array[adj_idx[i]]);
            if(slope_denom == 0) {
                slope_denom = 0.001;
            }

            x_tmp = (y_int_array[adj_idx[i]] - y_int_array[opposite_idx]) / slope_denom;
            corners[3-i][0] = x_tmp;
            corners[3-i][1] = slope_array[opposite_idx] * x_tmp + y_int_array[opposite_idx];
        }

        Point rect[] = new Point[4];

        for(int i = 0; i < 4; i++) {
            rect[i] = new Point(corners[i]);
        }

        quadrangle.fromArray(rect);

        return;
    }



    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Log.d(TAG, "TrackerKCF onCameraFrame: "+ tracking_flag);

        List<MatOfPoint> img_contours = new ArrayList<>();
        Mat hierarchy = new Mat();

        double contour_area = 0.0;
        double max_area = 0.0;
        MatOfPoint max_hull_contour = new MatOfPoint();

        // Timer
        long start = Core.getTickCount();
        // Grab camera frame in rgba and grayscale format
        mRgba = inputFrame.rgba();
        // Grab camera frame in gray format
        mGray = inputFrame.gray();

        Imgproc.putText(mRgba, "Streaming!", new Point(mRgba.rows() - 25, 35), Core.FONT_HERSHEY_SIMPLEX, 1.5, new Scalar(0, 0, 255), 2);

        // Dilate and Canny Edge detect grayscale img
        Imgproc.dilate(mGray, mGray, Mat.ones(3,3,CvType.CV_8UC1));
        Imgproc.Canny(mGray, mGray, 100, 200, 3, false);

        // Get contours from edge image
        Imgproc.findContours(mGray, img_contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        // Find largest contour by area (close using convex hull)
        for(int i = 0; i < img_contours.size(); i++) {
            // Get convex hull of contour
            MatOfInt hull = new MatOfInt();
            Imgproc.convexHull(img_contours.get(i), hull);

            // Create list of points in convex hull (subset of contour points)
            Point[] cntr_pts = img_contours.get(i).toArray();
            List<Point> hull_pts = new ArrayList<Point>();
            for(int idx : hull.toArray()) {
                hull_pts.add(cntr_pts[idx]);
            }

            // Convert list of points to contour
            MatOfPoint hull_contour = new MatOfPoint();
            hull_contour.fromList(hull_pts);

            // Calculate hull contour area and overwrite if new max area
            contour_area = Imgproc.contourArea(hull_contour);
            if(max_area < contour_area) {
                max_hull_contour = hull_contour;
                max_area = contour_area;
            }
        }

        // Draw max hull contour
        Imgproc.drawContours(mRgba, Arrays.asList(max_hull_contour), 0, new Scalar(0, 255, 0), 2, Core.LINE_8, hierarchy, 0, new Point());

        MatOfPoint quadrangle = new MatOfPoint();
        quadrangle_approx(max_hull_contour, quadrangle);

        Imgproc.drawContours(mRgba, Arrays.asList(quadrangle), 0, new Scalar(0, 255, 0), 2, Core.LINE_8, hierarchy, 0, new Point());

        // Returned frame will be displayed on the screen
        return mRgba;
    }
}