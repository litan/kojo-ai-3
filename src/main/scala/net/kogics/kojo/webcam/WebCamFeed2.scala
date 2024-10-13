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
package net.kogics.kojo.webcam

// This is an experimental WebCam Feed capture-er based on
// an opencv videocapture vs a javacv framegrabber
// This does not currently work in kojo-ai due to a missing native library
// The idea with this is to have an exploration point available for video capture on the Mac,
// which does not currently work with a javacv framegraber

import org.bytedeco.javacv.Java2DFrameUtils
import org.bytedeco.javacv.OpenCVFrameConverter
import org.bytedeco.opencv.opencv_core.Mat
import org.opencv.core.{ Mat => OpenCvMat }
import org.opencv.videoio.VideoCapture

class WebCamFeed2(device: Int, fps: Int) {
  @volatile private var running = false

  def startCapture(frameHandler: Mat => Unit): Unit = {
    var lastFrameTime = System.currentTimeMillis()
    running = true

    def detectFrameSequence(grabber: VideoCapture): Unit = {
      val delay = 1000.0 / fps
//      grabber.start()
      try {
        val frame0 = new OpenCvMat()
        grabber.read(frame0)
        val converter = new OpenCVFrameConverter.ToMat()
        while (!frame0.empty() && running) {
          val frame = converter.convert(frame0)
          val currTime = System.currentTimeMillis()
          if (currTime - lastFrameTime > delay) {
            val imageMat = Java2DFrameUtils.toMat(frame)
            frameHandler(imageMat)
            lastFrameTime = currTime
            Thread.sleep(0)
          }
          grabber.read(frame0)
        }
      }
      catch {
        case t: Throwable => // ignore
      }
      finally {
        running = false
        grabber.release()
      }
    }

    val grabber = new VideoCapture(0)
    detectFrameSequence(grabber)
  }

  def stopCapture(): Unit = {
    println("Stopping Capture")
    running = false
  }
}
