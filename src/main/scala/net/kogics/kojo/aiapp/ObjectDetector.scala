package net.kogics.kojo.aiapp

import java.awt.image.BufferedImage
import java.nio.file.Paths

import scala.util.Using

import ai.djl.modality.cv.output.DetectedObjects
import ai.djl.modality.cv.Image
import ai.djl.repository.zoo.Criteria
import org.bytedeco.opencv.opencv_core.Mat

class ObjectDetector(modelDir: String) {
  val (model, predictor) = {
    println("Loading 'object detection' model...")
    val translatorFactory = modelDir match {
      case s if s.contains("yolov5") => new ai.djl.modality.cv.translator.YoloV5TranslatorFactory()
      case s if s.contains("yolov8") => new ai.djl.modality.cv.translator.YoloV8TranslatorFactory()
      case _                         => throw new RuntimeException("Unknown object detection model")
    }

    import scala.jdk.CollectionConverters._
    val args: Map[String, AnyRef] = Map(
      "width" -> Int.box(640),
      "height" -> Int.box(640),
      "resize" -> Boolean.box(true),
      "rescale" -> Boolean.box(true),
      "optApplyRatio" -> Boolean.box(true),
      "threshold" -> Double.box(0.6),
    )

    val criteria =
      Criteria
        .builder()
        .setTypes(classOf[Image], classOf[DetectedObjects])
        .optModelPath(Paths.get(modelDir))
        .optTranslatorFactory(translatorFactory)
        .optArguments(args.asJava)
        .build()

    val mdl = criteria.loadModel()
    println("Done")
    (mdl, mdl.newPredictor())
  }

  def findObjects(imageMat: Mat): (DetectedObjects, BufferedImage) = {
    import ai.djl.pytorch.jni.JniUtils
    JniUtils.setGraphExecutorOptimize(false)
    Using.Manager { use =>
      val djlImage = matToDjlImage(imageMat)
      val detectedObjects = predictor.predict(djlImage)
      djlImage.drawBoundingBoxes(detectedObjects)
      (detectedObjects, djlImage.getWrappedImage.asInstanceOf[BufferedImage])
    }.get
  }

  def close(): Unit = {
    predictor.close()
    model.close()
  }

}
