package com.example.facerecogniser;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {

    private static final int PICK_IMAGE = 100;
    private static final int EMOTION_INPUT = 64;
    private static final int GENDER_INPUT = 128;
    private static final int AGE_INPUT = 200;
    private static final String[] EMOTION_LABELS =
            {"Happy", "Sad", "Angry", "Surprised", "Neutral", "Disgust", "Fear"};
    private ImageView imageView;
    private TextView textResult;
    private Interpreter emotionInterpreter;
    private Interpreter ageInterpreter;
    private Interpreter genderInterpreter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        textResult = findViewById(R.id.textResult);
        Button buttonSelectImage = findViewById(R.id.buttonSelectImage);

        try {
            emotionInterpreter = new Interpreter(loadModelFile("emotion_model.tflite"));
            ageInterpreter = new Interpreter(loadModelFile("age_model.tflite"));
            genderInterpreter = new Interpreter(loadModelFile("gender_model.tflite"));
        } catch (IOException e) {
            e.printStackTrace();
        }

        buttonSelectImage.setOnClickListener(v -> pickImageFromGallery());
    }

    private void pickImageFromGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent, PICK_IMAGE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == PICK_IMAGE && resultCode == RESULT_OK && data != null) {
            Uri uri = data.getData();
            try {
                Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
                imageView.setImageBitmap(bitmap);
                analyzeImage(bitmap);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private void analyzeImage(Bitmap bitmap) {
        InputImage inputImage = InputImage.fromBitmap(bitmap, 0);

        // High-accuracy face detection
        FaceDetectorOptions options = new FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                .build();
        FaceDetector detector = FaceDetection.getClient(options);

        detector.process(inputImage)
                .addOnSuccessListener(faces -> {
                    if (!faces.isEmpty()) {
                        Face face = faces.get(0);
                        int imgW = bitmap.getWidth();
                        int imgH = bitmap.getHeight();
                        int left   = Math.max(0, face.getBoundingBox().left);
                        int top    = Math.max(0, face.getBoundingBox().top);
                        int right  = Math.min(imgW, face.getBoundingBox().right);
                        int bottom = Math.min(imgH, face.getBoundingBox().bottom);
                        int w = Math.max(1, right - left);
                        int h = Math.max(1, bottom - top);

                        Bitmap faceBitmap = Bitmap.createBitmap(bitmap, left, top, w, h);

                        String emotion = classifyEmotion(faceBitmap);
                        int age = predictAge(faceBitmap);
                        String gender = predictGender(faceBitmap);

                        textResult.setText("Emotion: " + emotion + "\nAge: " + age + "\nGender: " + gender);
                    } else {
                        Toast.makeText(this, "No face detected", Toast.LENGTH_SHORT).show();
                    }
                })
                .addOnFailureListener(Throwable::printStackTrace);
    }

    private String classifyEmotion(Bitmap faceBitmap) {
        if (emotionInterpreter == null) return "Unknown";

        Bitmap resized = Bitmap.createScaledBitmap(faceBitmap, EMOTION_INPUT, EMOTION_INPUT, true);
        ByteBuffer input = convertBitmapToByteBuffer(resized, EMOTION_INPUT);

        // output shape [1, 7]
        float[][] output = new float[1][EMOTION_LABELS.length];
        emotionInterpreter.run(input, output);

        int idx = argMax(output[0]);
        return EMOTION_LABELS[idx];
    }

    private int predictAge(Bitmap faceBitmap) {
        if (ageInterpreter == null) return -1;

        Bitmap resized = Bitmap.createScaledBitmap(faceBitmap, AGE_INPUT, AGE_INPUT, true);
        ByteBuffer input = convertBitmapToByteBuffer(resized, AGE_INPUT);

            float[][] outReg = new float[1][1];
            ageInterpreter.run(input, outReg);
            Log.d("AgeDebug", "Raw output = " + outReg[0][0]);
            return Math.round(outReg[0][0]*116);
    }

    private String predictGender(Bitmap faceBitmap) {
        if (genderInterpreter == null) return "Unknown";

        Bitmap resized = Bitmap.createScaledBitmap(faceBitmap, GENDER_INPUT, GENDER_INPUT, true);
        ByteBuffer input = convertBitmapToByteBuffer(resized, GENDER_INPUT);

        // output shape [1, 2] = [Male, Female]
        float[][] output = new float[1][2];
        genderInterpreter.run(input, output);

        int idx = argMax(output[0]); // 0 = Male, 1 = Female
        return (idx == 0) ? "Male" : "Female";
    }

    private int argMax(float[] arr) {
        int idx = 0;
        float max = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > max) { max = arr[i]; idx = i; }
        }
        return idx;
    }

    private MappedByteBuffer loadModelFile(String filename) throws IOException {
        try (android.content.res.AssetFileDescriptor afd = getAssets().openFd(filename);
             FileInputStream fis = new FileInputStream(afd.getFileDescriptor());
             FileChannel channel = fis.getChannel()) {
            long startOffset = afd.getStartOffset();
            long declaredLength = afd.getDeclaredLength();
            return channel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }
    }

    // Resize & normalize image for TFLite
    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap, int imageSize) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] pixels = new int[imageSize * imageSize];
        bitmap.getPixels(pixels, 0, imageSize, 0, 0, imageSize, imageSize);
        int pixelIndex = 0;
        for (int y = 0; y < imageSize; y++) {
            for (int x = 0; x < imageSize; x++) {
                int val = pixels[pixelIndex++];
                float r = ((val >> 16) & 0xFF) / 255.0f;
                float g = ((val >> 8) & 0xFF) / 255.0f;
                float b = (val & 0xFF) / 255.0f;
                byteBuffer.putFloat(r);
                byteBuffer.putFloat(g);
                byteBuffer.putFloat(b);
            }
        }
        return byteBuffer;
    }
}
