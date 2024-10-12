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

import org.bytedeco.javacpp.indexer.FloatIndexer
import org.bytedeco.opencv.global.opencv_core.CV_32F
import org.bytedeco.opencv.global.opencv_dnn.{blobFromImage, readNetFromCaffe}
import org.bytedeco.opencv.global.opencv_imgproc.rectangle
import org.bytedeco.opencv.opencv_core._

import java.io.File

class FaceDetector(modelDir: String) {
  val fdConfidenceThreshold = 0.5
  val fdModelConfiguration = new File(s"$modelDir/deploy.prototxt")
  val fdModelBinary = new File(s"$modelDir/res10_300x300_ssd_iter_140000.caffemodel")
  val inWidth = 300
  val inHeight = 300
  val inScaleFactor = 1.0
  val meanVal = new Scalar(104.0, 177.0, 123.0, 128)
  val markerColor = new Scalar(0, 255, 255, 0)

  require(
    fdModelConfiguration.exists(),
    s"Cannot find FD model configuration: ${fdModelConfiguration.getCanonicalPath}"
  )

  require(fdModelBinary.exists(), s"Cannot find FD model file: ${fdModelBinary.getCanonicalPath}")

  val faceDetectionModel = readNetFromCaffe(fdModelConfiguration.getCanonicalPath, fdModelBinary.getCanonicalPath)

  def locateAndMarkFaces(image: Mat): Seq[Rect] = {
    // We will need to scale results for display on the input image, we need its width and height
    val imageWidth = image.size(1)
    val imageHeight = image.size(0)

    // Convert image to format suitable for using with the net
    val inputBlob = blobFromImage(image, inScaleFactor, new Size(inWidth, inHeight), meanVal, false, false, CV_32F)

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
        if confidence > fdConfidenceThreshold
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

  def close (): Unit = {
    faceDetectionModel.close()
  }
}
