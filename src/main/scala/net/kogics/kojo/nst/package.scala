package net.kogics.kojo
import java.awt.image.BufferedImage

import ai.djl.modality.cv.BufferedImageFactory
import ai.djl.ndarray._
import ai.djl.ndarray.types._
import ai.djl.translate._

package object nst {

  case class NstInputData(
      content: BufferedImage,
      style: BufferedImage,
      alpha: Float
  )

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
