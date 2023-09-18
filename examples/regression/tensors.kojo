import ai.djl.ndarray._
import scala.util.Using

// y = 3 x^2
// dy/dx = 6 x
// at x = 4, y = 48, dy/dx = 24

def y(x: NDArray): NDArray = {
    x.pow(2).mul(3)
}

def newGradientCollector = {
    import ai.djl.engine.Engine
    Engine.getInstance.newGradientCollector()
}

Using.Manager { use =>
    val nd = use(NDManager.newBaseManager())
    val gc = use(newGradientCollector)
    val x1 = nd.create(4f)
    x1.setRequiresGradient(true)
    val y1 = y(x1)
    gc.backward(y1)
    println(x1)
    println(y1)
    println(x1.getGradient)
}.get
