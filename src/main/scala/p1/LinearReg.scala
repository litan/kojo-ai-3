package p1

import ai.djl.engine.Engine
import ai.djl.ndarray.types.Shape
import ai.djl.ndarray.{NDList, NDManager}

import scala.jdk.CollectionConverters._
import scala.util.Using

object LinearReg {
  val Random = new java.util.Random

  def randomDouble(upperBound: Double): Double = {
    if ((upperBound == 0) || (upperBound != upperBound)) 0
    else
      Random.nextDouble * upperBound
  }

  def randomDouble(lowerBound: Double, upperBound: Double): Double = {
    if (lowerBound >= upperBound) lowerBound
    else
      lowerBound + randomDouble(upperBound - lowerBound)
  }

  def main(args: Array[String]): Unit = {
    System.setProperty("ai.djl.default_engine", "PyTorch")
    run()
  }

  def run(): Unit = {
    val m = 20
    val c = 3
    val xData = Array.tabulate(20)(e => (e + 1.0))
    val yData = xData map (x => x * m + c + randomDouble(-0.9, 0.9))

    val xDataf = xData.map(_.toFloat)
    val yDataf = yData.map(_.toFloat)

    val model = new Model
    model.train(xDataf, yDataf)
    val yPreds = model.predict(xDataf)
    val yPreds0 = yPreds.map(_.toDouble)
    model.close()
  }

  class Model {
    val LEARNING_RATE: Float = 0.1f
    val nd = NDManager.newBaseManager()
    nd.getManagedArrays.asScala.foreach { nda =>
      println(s"initial array with shape - ${nda.getShape}")
    }

    val w = nd.create(1f).reshape(new Shape(1, 1))
    val b = nd.create(0f).reshape(new Shape(1))

    val params = new NDList(w, b).asScala

    params.foreach { p =>
      p.setRequiresGradient(true)
    }

    def newGradientCollector = {
      Engine.getInstance.newGradientCollector()
    }

    def train(xValues: Array[Float], yValues: Array[Float]): Unit = {
      val x = nd.create(xValues).reshape(new Shape(-1, 1))
      val y = nd.create(yValues).reshape(new Shape(-1, 1))
      try {
        for (epoch <- 1 to 500) {
          Using.Manager { use =>
            val gc = use(newGradientCollector)
            gc.zeroGradients()
            val yPred = use(x matMul w add b)
            val loss = use(yPred.sub(y).pow(2).mean())
            gc.backward(loss)
          }.get // force an exception if there was a problem

          params.zipWithIndex.foreach { case (p, i) =>
            p.subi(p.getGradient.mul(LEARNING_RATE).div(20))
          }
        }
      } catch {
        case t: Throwable =>
          println("Problem")
          println(t)
      }
      println("Training Done")
      println(w)
      println(b)
    }

    def predict(xValues: Array[Float]): Array[Float] = {
      val x = nd.create(xValues).reshape(new Shape(-1, 1))
      val y = x matMul w add b
      y.toFloatArray
    }

    def close(): Unit = {
      println("Closing remaining ndarrays...")
      w.close()
      b.close()
      nd.close()
      println("Done")
    }
  }
}
