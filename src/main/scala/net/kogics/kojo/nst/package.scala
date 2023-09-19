package net.kogics.kojo
import java.awt.image.BufferedImage
import java.nio.file.Paths
import java.util.concurrent.ConcurrentHashMap

import ai.djl.modality.cv.BufferedImageFactory
import ai.djl.ndarray._
import ai.djl.ndarray.types._
import ai.djl.repository.zoo.Criteria
import ai.djl.repository.zoo.ZooModel
import ai.djl.translate._

package object nst {
  case class NstInputData(
      content: BufferedImage,
      style: BufferedImage,
      alpha: Float
  )

  type Input = NstInputData
  type Output = BufferedImage
  private val nstModelCache = nn.modelCache.asInstanceOf[ConcurrentHashMap[String, ZooModel[Input, Output]]]

  def cachedModel(modelDir: String, engine: String): ZooModel[Input, Output] = {
    val cacheKey = s"${modelDir}_$engine"
    val translator = engine match {
      case "PyTorch"    => new PyTorchNstTranslator()
      case "TensorFlow" => new TfNstTranslator()
    }
    var model = nstModelCache.get(cacheKey)
    if (model == null) {
      val criteria =
        Criteria
          .builder()
          .setTypes(classOf[Input], classOf[Output])
          .optEngine(engine)
          .optTranslator(translator)
          .optModelPath(Paths.get(modelDir))
          .build()

      model = criteria.loadModel()
      println(s"Caching model: $cacheKey")
      nstModelCache.put(cacheKey, model)
    }
    model
  }

  def imageToNDArray(image: BufferedImage, nd: NDManager): NDArray = {
    val djlImage = new BufferedImageFactory().fromImage(image)
    djlImage.toNDArray(nd)
  }

  def ndArrayToImage(nda: NDArray): BufferedImage = {
    val r1 = new BufferedImageFactory().fromNDArray(nda.clip(0, 255))
    r1.getWrappedImage.asInstanceOf[BufferedImage]
  }

  class PyTorchNstTranslator extends Translator[NstInputData, BufferedImage] {

    def processInput(ctx: TranslatorContext, input: NstInputData): NDList = {
      val nd = ctx.getNDManager
      val content = imageToNDArray(input.content, nd)
        .toType(DataType.FLOAT32, false)
        .transpose(2, 0, 1)
      val style = imageToNDArray(input.style, nd)
        .toType(DataType.FLOAT32, false)
        .transpose(2, 0, 1)
      val alpha = nd.create(input.alpha)
      new NDList(content, style, alpha)
    }

    def processOutput(ctx: TranslatorContext, list: NDList): BufferedImage = {
      val out = list.get(0)
      ndArrayToImage(out)
    }
  }

  class TfNstTranslator extends Translator[NstInputData, BufferedImage] {
    def processInput(ctx: TranslatorContext, input: NstInputData): NDList = {
      val nd = ctx.getNDManager
      val content = imageToNDArray(input.content, nd).toType(DataType.FLOAT32, false)
      val style = imageToNDArray(input.style, nd).toType(DataType.FLOAT32, false)
      val alpha = nd.create(input.alpha)
      new NDList(content, style, alpha)
    }

    def processOutput(ctx: TranslatorContext, list: NDList): BufferedImage = {
      val out = list.get(0)
      ndArrayToImage(out)
    }
  }

}
