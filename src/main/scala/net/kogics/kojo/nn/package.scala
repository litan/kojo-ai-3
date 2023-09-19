package net.kogics.kojo

import ai.djl.repository.zoo.ZooModel

import java.util.concurrent.ConcurrentHashMap

package object nn {
  val modelCache = new ConcurrentHashMap[String, ZooModel[_, _]]()

}
