/*
 * Copyright (C) 2024 Lalit Pant <pant.lalit@gmail.com>
 *
 * The contents of this file are subject to the GNU General Public License
 * Version 3 (the "License"); you may not use this file
 * except in compliance with the License. You may obtain a copy of
 * the License at http://www.gnu.org/copyleft/gpl.html
 *
 * Software distributed under the License is distributed on an "AS
 * IS" basis, WITHOUT WARRANTY OF ANY KIND, either express or
 * implied. See the License for the specific language governing
 * rights and limitations under the License.
 *
 */
package net.kogics.kojo.aiapp

import ai.djl.modality.cv.Image
import ai.djl.modality.cv.transform._
import ai.djl.ndarray._
import ai.djl.translate._
import org.bytedeco.javacv.Java2DFrameUtils
import org.bytedeco.opencv.global.opencv_imgproc.resize
import org.bytedeco.opencv.opencv_core.{Mat, Rect, Size}

import java.io.File
import scala.util.Using

class FaceEmbedder(modelDir: String) {
  val scriptDir = "/home/lalit/work/ai_fundamentals/face-id"
  val facenetDir = modelDir
  val faceSize = 160

  require(new File(facenetDir).exists, s"Cannot find face model dir: ${facenetDir}")

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
    import ai.djl.modality.cv.Image
    import ai.djl.repository.zoo.Criteria

    import java.nio.file.Paths
    val criteria =
      Criteria
        .builder()
        .setTypes(classOf[Image], classOf[Array[Float]])
        .optTranslator(new FeatureModelTranslator())
        .optModelPath(Paths.get(facenetDir))
        .build()

    val mdl = criteria.loadModel()
    println("Done")
    (mdl, mdl.newPredictor())
  }

  def showModel(): Unit = {
    faceFeatureModel.toString
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

  def close(): Unit = {
    faceFeaturePredictor.close()
    faceFeatureModel.close()
  }
}
