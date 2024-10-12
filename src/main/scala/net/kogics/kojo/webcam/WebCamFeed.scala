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

import org.bytedeco.javacv.FrameGrabber
import org.bytedeco.javacv.Java2DFrameUtils
import org.bytedeco.javacv.OpenCVFrameGrabber
import org.bytedeco.opencv.opencv_core.Mat

class WebCamFeed {
  @volatile private var running = false

  def startCapture(frameHandler: Mat => Unit): Unit = {
    var lastFrameTime = System.currentTimeMillis()
    running = true

    def detectFrameSequence(grabber: FrameGrabber): Unit = {
      val fps = 10
      val delay = 1000.0 / fps
      grabber.start()
      try {
        var frame = grabber.grab()
        while (frame != null && running) {
          val currTime = System.currentTimeMillis()
          if (currTime - lastFrameTime > delay) {
            val imageMat = Java2DFrameUtils.toMat(frame)
            frameHandler(imageMat)
            lastFrameTime = currTime
            Thread.sleep(0)
          }
          frame = grabber.grab()
        }
      }
      catch {
        case t: Throwable => // ignore
      }
      finally {
        running = false
        grabber.stop()
      }
    }

    val grabber = new OpenCVFrameGrabber(0)
    detectFrameSequence(grabber)
  }

  def stopCapture(): Unit = {
    println("Stopping Capture")
    running = false
  }
}
