// scroll down to the bottom of the file
// for code to play with

cleari()
clearOutput()

import java.io.File
import java.awt.image.BufferedImage
import java.util.HashMap

import scala.util.Using

import org.bytedeco.javacv._
import org.bytedeco.opencv.global.opencv_core.CV_32F
import org.bytedeco.opencv.global.opencv_dnn.{ readNetFromCaffe, blobFromImage }
import org.bytedeco.opencv.opencv_core.{ Rect, Mat, Size, Scalar, Point }
import org.bytedeco.javacpp.indexer.FloatIndexer
import org.bytedeco.opencv.global.opencv_imgproc.rectangle
import org.bytedeco.opencv.global.opencv_imgproc.resize
import org.bytedeco.opencv.global.opencv_imgcodecs.imread

import net.kogics.kojo.util.Utils

// stuff for the face detection net
val scriptDir = Utils.kojoCtx.baseDir
val fdConfidenceThreshold = 0.5
val fdModelConfiguration = new File(s"$scriptDir/face_detection_model/deploy.prototxt")
val fdModelBinary = new File(s"$scriptDir/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")
val inWidth = 300
val inHeight = 300
val inScaleFactor = 1.0
val meanVal = new Scalar(104.0, 177.0, 123.0, 128)
val markerColor = new Scalar(0, 255, 255, 0)

// stuff for the face feature net
val facenetDir = s"$scriptDir/face_feature_model"
val faceSize = 160

require(
    fdModelConfiguration.exists(),
    s"Cannot find FD model configuration: ${fdModelConfiguration.getCanonicalPath}")

require(
    fdModelBinary.exists(),
    s"Cannot find FD model file: ${fdModelBinary.getCanonicalPath}")

require(
    new File(facenetDir).exists,
    s"Cannot find face model dir: ${facenetDir}")

val faceDetectionModel = readNetFromCaffe(fdModelConfiguration.getCanonicalPath, fdModelBinary.getCanonicalPath)

import ai.djl.translate._
import ai.djl.ndarray._
import ai.djl.modality.cv.Image
import ai.djl.modality.cv.transform._
class FeatureModelTranslator extends Translator[Image, Array[Float]] {
    def processInput(ctx: TranslatorContext, input: Image): NDList = {
        val array = input.toNDArray(ctx.getNDManager)
        val pipeline = new Pipeline()
        pipeline
            .add(new ToTensor())
            .add(
                new Normalize(
                    Array(127.5f / 255.0f, 127.5f / 255.0f, 127.5f / 255.0f),
                    Array(128.0f / 255.0f, 128.0f / 255.0f, 128.0f / 255.0f)
                )
            )
        pipeline.transform(new NDList(array))
    }

    def processOutput(ctx: TranslatorContext, list: NDList): Array[Float] = {
        list.singletonOrThrow().toFloatArray
    }
}

val (faceFeatureModel, faceFeaturePredictor) = {
    println("Loading 'face feature extraction' model...")
    import ai.djl.repository.zoo.Criteria
    import ai.djl.modality.cv.Image
    import java.nio.file.Paths
    val criteria =
        Criteria.builder()
            .setTypes(classOf[Image], classOf[Array[Float]])
            .optTranslator(new FeatureModelTranslator())
            .optModelPath(Paths.get(facenetDir))
            .build()

    val mdl = criteria.loadModel()
    println("Done")
    (mdl, mdl.newPredictor())
}

@volatile var videoFramePic: Picture = _
@volatile var lastFrameTime = epochTimeMillis
@volatile var currImageMat: Mat = _
@volatile var currFaces: Seq[Rect] = _

def detectSequence(grabber: FrameGrabber): Unit = {
    val delay = 1000.0 / fps
    grabber.start()
    try {
        var frame = grabber.grab()
        while (frame != null) {
            frame = grabber.grab()
            val currTime = epochTimeMillis
            if (currTime - lastFrameTime > delay) {
                val imageMat = Java2DFrameUtils.toMat(frame)
                if (imageMat != null) { // sometimes the first few frames are empty so we ignore them
                    val faces = locateAndMarkFaces(imageMat)
                    val vfPic2 = Picture.image(Java2DFrameUtils.toBufferedImage(imageMat))
                    centerPic(vfPic2, imageMat.size(1), imageMat.size(0))
                    vfPic2.draw()
                    currImageMat = imageMat
                    currFaces = faces
                    if (videoFramePic != null) {
                        videoFramePic.erase()
                    }
                    videoFramePic = vfPic2
                    lastFrameTime = currTime
                }
            }
        }
    }
    catch {
        case _ => // eat up interruption
    }
    finally {
        grabber.stop()
        scriptDone()
    }
}

def centerPic(pic: Picture, w: Int, h: Int) {
    pic.translate(-w / 2, -h / 2)
}

def locateAndMarkFaces(image: Mat): Seq[Rect] = {
    // We will need to scale results for display on the input image, we need its width and height
    val imageWidth = image.size(1)
    val imageHeight = image.size(0)

    // Convert image to format suitable for using with the net
    val inputBlob = blobFromImage(
        image, inScaleFactor, new Size(inWidth, inHeight), meanVal, false, false, CV_32F)

    // Set the network input
    faceDetectionModel.setInput(inputBlob)

    // Make forward pass, compute output
    val detections = faceDetectionModel.forward()

    // Decode detected face locations
    val di = detections.createIndexer().asInstanceOf[FloatIndexer]
    val faceRegions =
        for {
            i <- 0 until detections.size(2)
            confidence = di.get(0, 0, i, 2)
            if (confidence > fdConfidenceThreshold)
        } yield {
            val x1 = (di.get(0, 0, i, 3) * imageWidth).toInt
            val y1 = (di.get(0, 0, i, 4) * imageHeight).toInt
            val x2 = (di.get(0, 0, i, 5) * imageWidth).toInt
            val y2 = (di.get(0, 0, i, 6) * imageHeight).toInt
            new Rect(new Point(x1, y1), new Point(x2, y2))
        }

    for (rect <- faceRegions) {
        rectangle(image, rect, markerColor)
    }

    faceRegions
}

def faceEmbedding(image: Mat): Array[Float] = {
    import ai.djl.modality.cv.BufferedImageFactory
    import ai.djl.pytorch.jni.JniUtils

    JniUtils.setGraphExecutorOptimize(false)
    Using.Manager { use =>
        println("Calculating embedding")
        val src = Java2DFrameUtils.toBufferedImage(image)
        val faceImage = new BufferedImageFactory().fromImage(src)
        println(s"Feeding into net, img(${faceImage.getWidth}, ${faceImage.getHeight})")
        val ret = faceFeaturePredictor.predict(faceImage)
        println("Done")
        ret
    }.get
}

def extractAndResizeFace(imageMat: Mat, rect: Rect): Mat = {
    val faceMat = new Mat(imageMat, rect)
    resize(faceMat, faceMat, new Size(faceSize, faceSize))
    faceMat
}

def scriptDone() {
    println("Closing Models...")
    faceFeaturePredictor.close()
    faceFeatureModel.close()
    faceDetectionModel.close()
}

val btnWidth = 100
val gap = 10
val iRadius = 30

def button(label: String): Picture = {
    picStackCentered(
        Picture.rectangle(btnWidth, iRadius * 2).withFillColor(cm.lightBlue).withPenColor(gray),
        Picture.text(label).withPenColor(black)
    )
}

// -----------------------------
// Tweak stuff below as desired
// Some ideas:
// Tweak indicator colors
// Tweak threshold and explore false positives, false negatives, etc
// make buttons inactive while work is happening
// Learn an "average" face for better performance

val cb = canvasBounds
val learnButton = button("Learn")
val verifyButton = button("Verify")
val learnStatusIndicator = Picture.circle(iRadius).withPenColor(gray)
val verifyStatusIndicator = Picture.circle(iRadius).withPenColor(gray)

val panel =
    picRowCentered(
        learnStatusIndicator, Picture.hgap(gap),
        learnButton, Picture.hgap(gap), verifyButton,
        Picture.hgap(gap), verifyStatusIndicator
    )

draw(panel)
panel.setPosition(
    cb.x + (cb.width - panel.bounds.width) / 2 + iRadius,
    cb.y + iRadius + 5)

val similarityThreshold = 0.9
val fps = 5
var learnedEmbedding: Array[Float] = _

def distance(a: Array[Float], b: Array[Float]): Float = {
    var ret = 0.0
    var mod1 = 0.0
    var mod2 = 0.0
    val length = a.length;
    for (i <- 0 until length) {
        ret += a(i) * b(i)
        mod1 += a(i) * a(i)
        mod2 += b(i) * b(i)
    }

    ((ret / math.sqrt(mod1) / math.sqrt(mod2) + 1) / 2.0).toFloat
}

def checkFace(imageMat: Mat, faces: Seq[Rect]): Boolean = {
    if (currFaces.size == 1) {
        val faceMat = extractAndResizeFace(imageMat, faces(0))
        val emb = faceEmbedding(faceMat)
        val similarity = distance(emb, learnedEmbedding)
        println(s"Similarity - $similarity")
        if (similarity > similarityThreshold) true else false
    }
    else {
        println("Verification is done only if there is one face on the screen")
        false
    }
}

learnButton.onMouseClick { (x, y) =>
    if (currFaces.size == 1) {
        println("Learning Face")
        learnStatusIndicator.setFillColor(cm.purple)
        Utils.runAsyncQueued {
            val faceMat = extractAndResizeFace(currImageMat, currFaces(0))
            learnedEmbedding = faceEmbedding(faceMat)
            learnStatusIndicator.setFillColor(orange)
        }
    }
    else {
        println("There should be only one face on the screen for Learning")
        learnStatusIndicator.setFillColor(noColor)
        learnedEmbedding = null
    }

}

verifyButton.onMouseClick { (x, y) =>
    if (learnedEmbedding != null) {
        Utils.runAsyncQueued {
            val good = checkFace(currImageMat, currFaces)
            if (good) {
                verifyStatusIndicator.setFillColor(green)
            }
            else {
                verifyStatusIndicator.setFillColor(red)
            }
            Utils.schedule(1) {
                verifyStatusIndicator.setFillColor(noColor)
            }
        }
    }
    else {
        println("First Learn, then Verify!")
    }
}

val grabber = new OpenCVFrameGrabber(0)
Utils.runAsyncMonitored {
    detectSequence(grabber)
}
