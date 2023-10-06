package net.kogics.kojo

import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.ConcurrentHashMap

import scala.util.Using

import ai.djl.ndarray.BaseNDManager
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.NDScope
import ai.djl.pytorch.engine.PtGradientCollector
import ai.djl.pytorch.jni.JniUtils
import ai.djl.repository.zoo.ZooModel
import ai.djl.training.GradientCollector

package object nn {
  val modelCache = new ConcurrentHashMap[String, ZooModel[_, _]]()

  def ndMaker: NDManager = NDManager.newBaseManager()

  def managed[T](code: Using.Manager => T): T = {
    Using.Manager { use =>
      code(use)
    }.get
  }

  def ndScoped[T](code: Using.Manager => T): T = {
    managed { use =>
      use(new NDScope)
      code(use)
    }
  }

  def gradientCollector: GradientCollector = {
    import ai.djl.engine.Engine
    Engine.getInstance.newGradientCollector()
  }

  def resetGradientCollection(): Unit = {
    val cls = classOf[PtGradientCollector]
    val f = cls.getDeclaredField("isCollecting")
    f.setAccessible(true)
    f.get(cls).asInstanceOf[AtomicBoolean].set(false)
  }

  def ndDebugDump(ndManager: NDManager, level: Int): Unit = {
    ndManager.asInstanceOf[BaseNDManager].debugDump(level)
  }

  implicit class RichNDArray(nda: NDArray) {
    // +, -, *, /, **
    def zeroGradients(): Unit = {
      JniUtils.zeroGrad(nda.asInstanceOf[ai.djl.pytorch.engine.PtNDArray])
    }
  }
}
