package p1
import ai.djl.ndarray.{NDManager, NDScope}

import util.Using
object ScopePlay {
  def main(args: Array[String]): Unit = {
    Using.Manager { use =>
      val nd = use(NDManager.newBaseManager())
      val scope = use(new NDScope()); scope.suppressNotUsedWarning()
      val x1 = nd.create(4f)
      println(x1)
    }
  }
}
