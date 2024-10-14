package net.kogics.kojo

import java.awt.image.BufferedImage
import ai.djl.modality.cv.BufferedImageFactory
import ai.djl.modality.cv.Image
import org.bytedeco.javacv.Java2DFrameUtils
import org.bytedeco.opencv.opencv_core.Mat

package object aiapp {
  val biFactory = new BufferedImageFactory()

  def bufferedImageToDjlImage(bufImage: BufferedImage): Image = {
    biFactory.fromImage(bufImage)
  }

  def matToBufferedImage(imageMat: Mat): BufferedImage = {
    Java2DFrameUtils.toBufferedImage(imageMat)
  }

  def matToDjlImage(imageMat: Mat): Image = {
    bufferedImageToDjlImage(matToBufferedImage(imageMat))
  }
}
