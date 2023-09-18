/*
 * Copyright (C) 2019 Lalit Pant <pant.lalit@gmail.com>
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
package net.kogics.kojo

import scala.collection.mutable
import scala.collection.mutable.{ Map => MMap }
import scala.collection.mutable.{ Set => MSet }
import scala.collection.mutable.ArrayBuffer

package object graph {
  trait Container[T] {
    def add(t: T, cost: Double = 0): Unit
    def remove(): T
    def hasElements: Boolean
  }

  class Stack[T] extends Container[T] {
    var impl: List[T] = Nil
    def add(t: T, cost: Double = 0): Unit = {
      impl = t :: impl
    }
    def remove(): T = {
      val ret = impl.head
      impl = impl.tail
      ret
    }
    def hasElements = !impl.isEmpty
  }

  class Queue[T] extends Container[T] {
    val impl = collection.mutable.Queue.empty[T]
    def add(t: T, cost: Double = 0): Unit = {
      impl.enqueue(t)
    }
    def remove(): T = {
      impl.dequeue()
    }
    def hasElements = !impl.isEmpty
  }

  class PriorityQueue[T] extends Container[T] {
    case class TwithCost(t: T, cost: Double)
    implicit val TwithCostOrdering = new Ordering[TwithCost] {
      override def compare(x: TwithCost, y: TwithCost): Int = {
        -x.cost.compare(-y.cost)
      }
    }
    val impl = collection.mutable.PriorityQueue.empty[TwithCost]
    def add(t: T, cost: Double): Unit = {
      //      println(s"adding: $t with cost $cost")
      impl.enqueue(TwithCost(t, cost))
    }
    def remove(): T = {
      val head = impl.dequeue()
      //      println(s"removing: ${head.t} with cost ${head.cost}")
      head.t
    }
    def hasElements = !impl.isEmpty
  }

  case class Node[T](data: T)
  case class EdgeTo[T](node: Node[T], distance: Double = 1)

  type GraphEdges[T] = MMap[Node[T], MSet[EdgeTo[T]]]
  type Nodes[T] = MSet[Node[T]]
  type PathEdges[T] = mutable.Seq[EdgeTo[T]]
  type ContainerElem[T] = (Node[T], PathEdges[T])

  trait Graph[T] {
    def nodes: Nodes[T]
    def edges: GraphEdges[T]
  }

  case class ExplicitGraph[T](nodes: Nodes[T], edges: GraphEdges[T]) extends Graph[T]

  object GraphSearch {
    type CostFn[T] = (PathEdges[T], Node[T]) => Double
    type HeuristicFn[T] = (Node[T], Node[T]) => Double

    def noOpCallback[T](n: Node[T]): Unit = {}
    def dfs[T](
        graph: Graph[T],
        start: Node[T],
        end: Node[T],
        visitCallback: Node[T] => Unit = noOpCallback _
    ): Option[PathEdges[T]] = {
      def cost(pathEdges: PathEdges[T], end: Node[T]): Double = 0
      val container = new Stack[ContainerElem[T]]
      searchWithCost(graph, start, end, container, visitCallback, cost)
    }

    def bfs[T](
        graph: Graph[T],
        start: Node[T],
        end: Node[T],
        visitCallback: Node[T] => Unit = noOpCallback _
    ): Option[PathEdges[T]] = {
      def cost(pathEdges: PathEdges[T], end: Node[T]): Double = 0
      val container = new Queue[ContainerElem[T]]
      searchWithCost(graph, start, end, container, visitCallback, cost)
    }

    private def pathDistance[T](pathEdges: PathEdges[T]): Double = pathEdges.foldLeft(0.0) {
      case (d, e) => d + e.distance
    }

    def ucs[T](
        graph: Graph[T],
        start: Node[T],
        end: Node[T],
        visitCallback: Node[T] => Unit = noOpCallback _
    ): Option[PathEdges[T]] = {
      def cost(pathEdges: PathEdges[T], end: Node[T]): Double = pathDistance(pathEdges)
      val container = new PriorityQueue[ContainerElem[T]]
      searchWithCost(graph, start, end, container, visitCallback, cost)
    }

    def astarSearch[T](
        graph: Graph[T],
        start: Node[T],
        end: Node[T],
        visitCallback: Node[T] => Unit = noOpCallback _,
        heuristic: HeuristicFn[T]
    ): Option[PathEdges[T]] = {
      def cost(pathEdges: PathEdges[T], end: Node[T]): Double =
        pathDistance(pathEdges) + heuristic(pathEdges.last.node, end)
      val container = new PriorityQueue[ContainerElem[T]]
      searchWithCost(graph, start, end, container, visitCallback, cost)
    }

    def searchWithCost[T](
        graph: Graph[T],
        start: Node[T],
        end: Node[T],
        container: Container[ContainerElem[T]],
        visitCallback: Node[T] => Unit,
        costFn: CostFn[T]
    ): Option[PathEdges[T]] = {
      if (start == end) {
        Some(ArrayBuffer())
      }
      else {
        container.add((start, ArrayBuffer()))
        val visited = new collection.mutable.HashSet[Node[T]]
        while (container.hasElements) {
          val elem = container.remove()
          val elemNode = elem._1
          val elemPath = elem._2
          if (!visited.contains(elemNode)) {
            visitCallback(elemNode)
            visited.add(elemNode)
            if (elemNode == end) {
              return Some(elemPath)
            }
            else {
              graph.edges(elem._1).foreach { edgeTo =>
                val newPath = elemPath :+ edgeTo
                container.add((edgeTo.node, newPath), costFn(newPath, end))
              }
            }
          }
        }
        None
      }
    }
  }
}
