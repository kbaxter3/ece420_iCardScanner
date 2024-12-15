package com.ece420.lab7;


import android.app.ActionBar;
import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.Manifest;
import android.graphics.Bitmap;
import android.os.Bundle;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;

import com.googlecode.tesseract.android.TessBaseAPI;

import org.opencv.android.BaseLoaderCallback;

import org.opencv.android.CameraBridgeViewBase;

import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;

import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.lang.Math;
import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;
import java.util.stream.IntStream;

public class MainActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "MainActivity";
    private static final int W_PERSPECTIVE = 480;
    private static final int H_PERSPECTIVE = (int) (W_PERSPECTIVE / 1.6);
//    private static final MatOfPoint MAT_PERSPECTIVE = new MatOfPoint(new Point(0,0), new Point(0, H_PERSPECTIVE), new Point(W_PERSPECTIVE, H_PERSPECTIVE), new Point(W_PERSPECTIVE, 0));

    // UI Variables
    private Button controlButton;
    private Button clipboardButton;
    private String nameText;
    private String text;

    // Declare OpenCV based camera view base
    private CameraBridgeViewBase mOpenCvCameraView;

    // Mat to store RGBA and Grayscale camera preview frame
    private Mat mRgba;
    private Mat mGray;
    private Mat mWarpPersp;

    private int tracking_flag = -1;

    private TessBaseAPI tess;

    private int matches_prev_counter;
    private String prev_uin;
    private long start_time; // for testing purposes

    // https://javapapers.com/android/android-read-csv-file/
    public class CSVFile {
        InputStream inputStream;

        public CSVFile(InputStream inputStream){
            this.inputStream = inputStream;
        }

        public List read(){
            List resultList = new ArrayList();
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
            try {
                String csvLine;
                while ((csvLine = reader.readLine()) != null) {
                    String[] row = csvLine.split(",");
                    resultList.add(row);
                }
            }
            catch (IOException ex) {
                throw new RuntimeException("Error in reading CSV file: "+ex);
            }
            finally {
                try {
                    inputStream.close();
                }
                catch (IOException e) {
                    throw new RuntimeException("Error while closing input stream: "+e);
                }
            }
            return resultList;
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);
        super.setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);

        // To hide action bar
        try {
            ActionBar actionBar = getActionBar();
            actionBar.hide();
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Request User Permission on Camera
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED){
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA}, 1);}

        // OpenCV Loader and Avoid using OpenCV Manager
        if (!OpenCVLoader.initDebug()) {
            Log.e(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), not working.");
        } else {
            Log.d(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), working.");
        }

        // Setup control button
        controlButton = (Button)findViewById((R.id.controlButton));
        controlButton.setText("Pause");
        controlButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(tracking_flag == 1) {
                    tracking_flag = 0;
                    controlButton.setText("Pause");

                    // Timer
                    start_time = Core.getTickCount();

                } else {
                    tracking_flag = 1;
                    controlButton.setText("Resume");
                }
            }
        });


        // Setup clipboard button
        clipboardButton = (Button)findViewById((R.id.clipboardButton));
        clipboardButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Copy to clipboard (Chat GPT)
                ClipboardManager clipboard = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
                ClipData clip = ClipData.newPlainText("label", nameText + '\n'+  text);
                // Set the clip data to the clipboard
                clipboard.setPrimaryClip(clip);
            }
        });


        // Setup OpenCV Camera View
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.opencv_camera_preview);
        // Use main camera with 0 or front camera with 1
        mOpenCvCameraView.setCameraIndex(0);

        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);

        // Force camera resolution, ignored since OpenCV automatically select best ones
        mOpenCvCameraView.setCvCameraViewListener(this);

        // Declare string to store previous UIN
        prev_uin = "";
        matches_prev_counter = 0;

        // Initialize Tesseract API
        tess = new TessBaseAPI();

        // Given path must contain subdirectory `tessdata` where are `*.traineddata` language files
        // The path must be directly readable by the app
        try{
            // Path to the app's internal storage directory
            String tesspath = getFilesDir() + "/tessdata/";

            // Create tessdata directory if it doesn't exist
            File dir = new File(tesspath);
            if (!dir.exists()) {
                dir.mkdirs();
            }

            // File path for the language traineddata file
            String tessFilepath = tesspath + "eng" + ".traineddata";

            // Check if the file already exists
            File file = new File(tessFilepath);
            if (!file.exists()) {
                InputStream input = getAssets().open("tessdata/" + "eng" + ".traineddata");
                OutputStream output = new FileOutputStream(tessFilepath);

                // Copy the file
                byte[] buffer = new byte[1024];
                int length;
                while ((length = input.read(buffer)) > 0) {
                    output.write(buffer, 0, length);
                }

                // Close streams
                output.flush();
                output.close();
                input.close();
            }

            if (!tess.init(getFilesDir().getPath(), "eng")) { // could be multiple languages, like "eng+deu+fra"
                // Error initializing Tesseract (wrong/inaccessible data path or not existing language file(s))
                Log.d(TAG, "[ERROR] wrong/inaccessible traineddata file: " + tessFilepath);
                // Release the native Tesseract instance
                tess.recycle();
            }

        } catch (Exception e)
        {
            e.printStackTrace();
            Log.d(TAG, "[ERROR] can't load tessDataPath: " + e.getMessage());
        }

        Log.d(TAG, "[INFO] tess initialized");

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
        return;
    }

    // OpenCV Camera Functionality Code
    @Override
    public void onCameraViewStarted(int width, int height) {
        Log.d(TAG, "onCameraViewStarted");
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);
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

        // Angle between two slopes - https://www.cuemath.com/geometry/angle-between-two-lines/
        // Approx arctan as y=x, only need for ranking so don't need exact value, just monotonic increase
        int opposite_idx = 1;
        double slope_ratio = Math.abs((slope_array[1] - slope_array[0]) / (1 + slope_array[1]*slope_array[0]) ); // (m1-m2) / (1+m1m2)

        for(int i = 1; i < 4; ++i) {
            double new_slope_ratio = Math.abs((slope_array[i] - slope_array[0]) / (1 + slope_array[i]*slope_array[0]) );

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

    private String threshAndOCR(Mat croppedImage) {
        Imgproc.threshold(croppedImage, croppedImage, 0, 255, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);

        // https://stackoverflow.com/questions/13134682/convert-mat-to-bitmap-opencv-for-android
        Bitmap bmp = null;
        try {
            bmp = Bitmap.createBitmap(croppedImage.cols(), croppedImage.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(croppedImage, bmp);
        }
        catch (CvException e){Log.d("Exception", e.getMessage());}

        tess.setImage(bmp);
        return tess.getUTF8Text();
    }

    private class UpdateUITextTask implements Runnable {
        String str1;
        String str2;
        String str3;

        UpdateUITextTask(String s1,String s2,String s3) {str1 = s1; str2 = s2; str3 = s3;}

        public void run() {
            TextView TextView1 = (TextView) findViewById(R.id.labAccessTextView);
            TextView1.setText(str1);
            TextView TextView2 = (TextView) findViewById(R.id.uinTextView);
            TextView2.setText(str2);
            TextView TextView3 = (TextView) findViewById(R.id.nameTextView);
            TextView3.setText(str3);
        }
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        String labAccessMessage = "Lab Access: ";
        final MatOfPoint2f MAT_PERSPECTIVE = new MatOfPoint2f(new Point(0,0), new Point(0, H_PERSPECTIVE), new Point(W_PERSPECTIVE, H_PERSPECTIVE), new Point(W_PERSPECTIVE, 0));

        Log.d(TAG, "onCameraFrame: "+ tracking_flag);

        // Freeze Video feed when UIN found
        if(tracking_flag == 1) {
            return mRgba;
        }

        List<MatOfPoint> img_contours = new ArrayList<>();
        Mat hierarchy = new Mat();

        double contour_area = 0.0;
        double max_area = 0.0;
        MatOfPoint max_hull_contour = new MatOfPoint();

        // Grab camera frame in rgba and grayscale format
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray().clone();

        // Dilate and Canny Edge detect grayscale img
        Imgproc.dilate(mGray, mGray, Mat.ones(3,3,CvType.CV_8UC1));
        Imgproc.Canny(mGray, mGray, 100, 200, 3, false);

///////////// GET LARGEST CONTOUR (CONVEX-HULL AREA)
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

        if(max_area <= 0) {
            return mRgba;
        }

        // Draw max hull contour
        Imgproc.drawContours(mRgba, Arrays.asList(max_hull_contour), 0, new Scalar(0, 255, 0), 2, Core.LINE_8, hierarchy, 0, new Point());

///////////// GET BOUNDING QUADRANGLE APPROXIMATION OF LARGEST CONTOUR
        MatOfPoint quadrangle = new MatOfPoint();
        quadrangle_approx(max_hull_contour, quadrangle);

        Imgproc.drawContours(mRgba, Arrays.asList(quadrangle), 0, new Scalar(0, 255, 0), 2, Core.LINE_8, hierarchy, 0, new Point());

///////////// ORIENT QUADRANGLE POINTS - Orient points to be CW, short edge first
        Point ptArr[] = quadrangle.toArray();
        Point vec1, vec2;

        vec1 = new Point(ptArr[0].x - ptArr[1].x, ptArr[0].y - ptArr[1].y);
        vec2 = new Point(ptArr[0].x - ptArr[3].x, ptArr[0].y - ptArr[3].y);

        // Get CW/CCW by cross product magnitude
        if(vec1.x * vec2.y - vec1.y * vec2.x > 0) {
            // If CCW (Positive crossproduct between vec1 x vec2)
            // Reverse points to make it Clockwise
            for(int i = 0; i < ptArr.length / 2; i++) {
                Point tmp = ptArr[i];
                ptArr[i] = ptArr[ptArr.length - i - 1];
                ptArr[ptArr.length - i - 1] = tmp;
            }
        }

        vec1 = new Point(ptArr[0].x - ptArr[1].x, ptArr[0].y - ptArr[1].y);
        vec2 = new Point(ptArr[0].x - ptArr[3].x, ptArr[0].y - ptArr[3].y);

        // Check short edge first, by comparing vector magnitudes
        // if not, roll the pts so short edge is first
        if( (vec1.x * vec1.x + vec1.y * vec1.y) > (vec2.x * vec2.x + vec2.y * vec2.y) ) {
            Point tmp = ptArr[ptArr.length - 1];

            for (int i = ptArr.length - 1; i > 0; i--) {
                ptArr[i] = ptArr[i - 1];
            }

            ptArr[0] = tmp;
        }

        // Write back into quadrangle Mat
        quadrangle.fromArray(ptArr);

///////////// PERSPECTIVE TRANSFORM
        // Find Perspective Transform
        MatOfPoint2f quadrangle_2f = new MatOfPoint2f(quadrangle.toArray());
        Mat M_perspective = Imgproc.getPerspectiveTransform(quadrangle_2f, MAT_PERSPECTIVE);

        // Apply Perpsective Transform
        mWarpPersp = new Mat(mGray.height(), mGray.width(), CvType.CV_8UC1);
        Mat mWarpColor = new Mat();
        Imgproc.warpPerspective(mRgba, mWarpColor, M_perspective, mRgba.size());
        Imgproc.warpPerspective(inputFrame.gray(), mWarpPersp, M_perspective, mGray.size());

///////////// CHECK FOR 180 ROTATION AND CROP TO UIN & NAME
        // Orient with the orange stripe on the left
        double left_red_sum = 0;
        double right_red_sum = 0;

        for(int i = 0; i < W_PERSPECTIVE / 8; i++) {
            for(int j = 0; j < H_PERSPECTIVE; j++) {
                left_red_sum += mWarpColor.get(i,j)[0];
                right_red_sum += mWarpColor.get(W_PERSPECTIVE - i,j)[0];
            }
        }

        // Crop UIN/name area based on card rotation
        Mat croppedImage, nameCropImg;
        int x_start, y_start, roi_width, roi_height;
        int name_x_start, name_y_start, name_roi_width, name_roi_height;
        Rect uin_roi, name_roi;

        if(left_red_sum < right_red_sum) {
            // Crop Perspective Transformed Image
            x_start = (int)(W_PERSPECTIVE * (1 - 20 / 100 - 1/4) );
            y_start = (int)(0);
            roi_width = (int)(W_PERSPECTIVE / 4);
            roi_height = (int)(H_PERSPECTIVE / 10);
            uin_roi = new Rect(x_start, y_start, roi_width, roi_height);

            name_x_start = (int)(W_PERSPECTIVE * (1 - 13 / 100 - 8/10) );
            name_y_start = (int)(H_PERSPECTIVE * (1 - 73/100 - 1/10) );
            name_roi_width = (int)(W_PERSPECTIVE * 8 / 10);
            name_roi_height = (int)(H_PERSPECTIVE / 10);
            name_roi = new Rect(name_x_start, name_y_start, name_roi_width, name_roi_height);

            croppedImage = new Mat(mWarpPersp, uin_roi);
            nameCropImg  = new Mat(mWarpPersp, name_roi);

            Core.rotate(croppedImage, croppedImage, Core.ROTATE_180);
            Core.rotate(nameCropImg, nameCropImg, Core.ROTATE_180);
        } else {
            // Crop Perspective Transformed Image
            x_start = (int)(W_PERSPECTIVE * 20 / 100);
            y_start = (int)(H_PERSPECTIVE  - (H_PERSPECTIVE / 10));
            roi_width = (int)(W_PERSPECTIVE / 4);
            roi_height = (int)(H_PERSPECTIVE / 10);
            uin_roi = new Rect(x_start, y_start, roi_width, roi_height);

            name_x_start = (int)(W_PERSPECTIVE * 13 / 100 );
            name_y_start = (int)(H_PERSPECTIVE * 73/100 );
            name_roi_width = (int)(W_PERSPECTIVE * 8 / 10);
            name_roi_height = (int)(H_PERSPECTIVE / 10);
            name_roi = new Rect(name_x_start, name_y_start, name_roi_width, name_roi_height);

            croppedImage = new Mat(mWarpPersp, uin_roi);
            nameCropImg  = new Mat(mWarpPersp, name_roi);
        }

///////////// THRESHOLD IMAGE AND RUN OCR
        text = threshAndOCR(croppedImage);
        nameText = threshAndOCR(nameCropImg);

        nameText = nameText.replaceAll("^[^A-Z]*|[^A-Z]*$", "");
        Log.d(TAG, "Name output: " + nameText);

        // Check if valid UIN
        boolean isUIN = false;
        if( text.length() == 9 ) {
            isUIN = true;
            for (int i = 0; i < text.length(); i++) {
                if (!Character.isDigit(text.charAt(i))) {
                    isUIN = false;
                    break;
                }
            }
        }

        if(isUIN) {
            if(prev_uin.equals(text)) {
                matches_prev_counter++;
            } else {
                matches_prev_counter = 0;
                prev_uin = text;
            }

            if(matches_prev_counter >= 2) {
                // Record time to detection for testing
                double time_to_detect = (Core.getTickCount() - start_time) / Core.getTickFrequency();
                Imgproc.putText(mRgba, "Time: " + time_to_detect, new Point(10, 60), Core.FONT_HERSHEY_SIMPLEX, 1, new Scalar(0, 255, 0), 2);

                controlButton.setText("Rescan");
                tracking_flag = 1;
                matches_prev_counter = 0;
                prev_uin = "";

                // Check against database
                boolean has_access = false;
                InputStream is = getResources().openRawResource(R.raw.lab_access_log);
                CSVFile csv_file = new CSVFile(is);

                List list = csv_file.read();
                String[] l;
                for (int i = 0; i < list.size(); i++) {
                    l = (String[]) list.get(i);

                    for (String s : l) {
                        if (s.equals(text)) {
                            has_access = true;
                            break;
                        }
                    }
                }

                labAccessMessage = (has_access) ? "Lab Access: Yes" : "Lab Access: No";
            }
        }

        // Update UI text and update screen
        MainActivity.this.runOnUiThread(new UpdateUITextTask(labAccessMessage, "UIN: " + text, "Name: " + nameText));
        return mRgba;
    }
}


