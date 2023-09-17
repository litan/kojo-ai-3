package p1

import ai.djl.ndarray.NDManager

import scala.util.Using

object Main {
  def main(args: Array[String]): Unit = {
//    JniUtils.setGradMode(true)
//    JniUtils.setGraphExecutorOptimize(false)
    // y = 3 x^2
    // dy/dx = 6 x
    // at x = 2, dy/dx = 12

    def newGradientCollector = {
      import ai.djl.engine.Engine
      Engine.getInstance.newGradientCollector()
    }

    Using.Manager { use =>
      val nda = use(NDManager.newBaseManager())
      val x = nda.create(2f)
      x.setRequiresGradient(true)
      val y = x pow 2 mul 3
      //    val gc = use(newGradientCollector)
      //    gc.backward(y)
      println(x)
      println(y)
    //    println(x.getGradient)
    }
  }
}
