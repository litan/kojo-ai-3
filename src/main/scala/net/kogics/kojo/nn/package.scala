package net.kogics.kojo

import ai.djl.ndarray.NDArray
import ai.djl.pytorch.jni.JniUtils
import ai.djl.repository.zoo.ZooModel

import java.util.concurrent.ConcurrentHashMap

package object nn {
  val modelCache = new ConcurrentHashMap[String, ZooModel[_, _]]()

  implicit class RichNDArray(nda: NDArray) {
    def zeroGradients(): Unit = {
      JniUtils.zeroGrad(nda.asInstanceOf[ai.djl.pytorch.engine.PtNDArray])
    }
  }
}
