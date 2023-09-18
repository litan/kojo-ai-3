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

import java.util

import org.knowm.xchart._
import org.knowm.xchart.style.Styler.ChartTheme
import org.knowm.xchart.XYSeries.XYSeriesRenderStyle

package object plot {
  private def makeXYChart(title: String, xtitle: String, ytitle: String): XYChart = {
    val chart = new XYChartBuilder()
      .width(800)
      .height(600)
      .title(title)
      .xAxisTitle(xtitle)
      .yAxisTitle(ytitle)
      .theme(ChartTheme.GGPlot2)
      .build()
    chart.getStyler.setDefaultSeriesRenderStyle(XYSeriesRenderStyle.Scatter)
    chart.getStyler.setChartTitleVisible(true)
    chart.getStyler.setAxisTitlesVisible(true)
    chart.getStyler.setXAxisDecimalPattern("0.0")
    chart.getStyler.setYAxisDecimalPattern("0.0")
    chart.getStyler.setXAxisMin(0)
    chart.getStyler.setYAxisMin(0)
    chart
  }

  private def makeCategoryChart(title: String, xtitle: String, ytitle: String): CategoryChart = {
    val chart = new CategoryChartBuilder()
      .width(800)
      .height(600)
      .title(title)
      .xAxisTitle(xtitle)
      .yAxisTitle(ytitle)
      .theme(ChartTheme.GGPlot2)
      .build()
    chart.getStyler.setChartTitleVisible(true)
    chart.getStyler.setAxisTitlesVisible(true)
    chart
  }

  private def makePieChart(title: String): PieChart = {
    val chart = new PieChartBuilder().width(800).height(600).title(title).theme(ChartTheme.GGPlot2).build()
    chart.getStyler.setChartTitleVisible(true)
    chart
  }

  private def addXYSeriesToChart(
      chart: XYChart,
      lineName: Option[String],
      xs: Array[Double],
      ys: Array[Double]
  ): XYSeries = {
    lineName match {
      case Some(name) =>
        chart.getStyler.setLegendVisible(true)
        chart.addSeries(name, xs, ys)
      case None =>
        chart.getStyler.setLegendVisible(false)
        chart.addSeries("default", xs, ys)
    }
  }

  private def addCategorySeriesToChart[N <: Number](
      chart: CategoryChart,
      name: Option[String],
      categories: Seq[String],
      values: Seq[N]
  ): CategorySeries = {
    def asJavaList[A, B](xs: Seq[B]) = {
      import scala.jdk.CollectionConverters._
      xs.asJava.asInstanceOf[java.util.List[A]]
    }

    name match {
      case Some(name) =>
        chart.getStyler.setLegendVisible(true)
        chart.addSeries(name, asJavaList(categories), asJavaList(values))
      case None =>
        chart.getStyler.setLegendVisible(false)
        chart.addSeries("default", asJavaList(categories), asJavaList(values))
    }
  }

  private def addPieSlicesToChart(chart: PieChart, categories: Seq[String], counts: Seq[Int]): PieChart = {
    chart.getStyler.setLegendVisible(true)
    categories.zip(counts).foreach {
      case (cat, cnt) =>
        chart.addSeries(cat, cnt)
    }
    chart
  }

  private def addHistogramSeriesToChart(
      chart: CategoryChart,
      name: Option[String],
      xs: Array[Double],
      bins: Int
  ): CategorySeries = {
    def asJavaCollection(xs: Array[Double]) = {
      val al = new util.ArrayList[java.lang.Double]
      xs.foreach(al.add(_))
      al
    }

    chart.getStyler.setOverlapped(false)
    chart.getStyler.setXAxisDecimalPattern("0")
    chart.getStyler.setYAxisDecimalPattern("0")
    val histogram = new Histogram(asJavaCollection(xs), bins)

    name match {
      case Some(name) =>
        chart.getStyler.setLegendVisible(true)
        chart.addSeries(name, histogram.getxAxisData(), histogram.getyAxisData())
      case None =>
        chart.getStyler.setLegendVisible(false)
        chart.addSeries("default", histogram.getxAxisData(), histogram.getyAxisData())
    }
  }

  def addLineToChart(chart: XYChart, lineName: Option[String], xs: Array[Double], ys: Array[Double]): XYChart = {
    val series = addXYSeriesToChart(chart, lineName, xs, ys)
    series.setXYSeriesRenderStyle(XYSeriesRenderStyle.Line)
    chart
  }

  def addPointsToChart(chart: XYChart, lineName: Option[String], xs: Array[Double], ys: Array[Double]): XYChart = {
    val series = addXYSeriesToChart(chart, lineName, xs, ys)
    series.setXYSeriesRenderStyle(XYSeriesRenderStyle.Scatter)
    chart
  }

  def addBarsToChart[N <: Number](
      chart: CategoryChart,
      name: Option[String],
      categories: Seq[String],
      values: Seq[N]
  ): CategoryChart = {
    val series = addCategorySeriesToChart(chart, name, categories, values)
    //    series.setLineColor()
    chart
  }

  def addSlicesToChart(chart: PieChart, name: Option[String], categories: Seq[String], counts: Seq[Int]): PieChart = {
    val series = addPieSlicesToChart(chart, categories, counts)
    //    series.setLineColor()
    chart
  }

  def addHistogramToChart(chart: CategoryChart, name: Option[String], xs: Array[Double], bins: Int): CategoryChart = {
    val series = addHistogramSeriesToChart(chart, name, xs, bins)
    //    series.setLineColor()
    chart
  }

  def scatterChart(title: String, xtitle: String, ytitle: String, xs: Array[Double], ys: Array[Double]): XYChart = {
    val chart = makeXYChart(title, xtitle, ytitle)
    addPointsToChart(chart, None, xs, ys)
  }

  def lineChart(title: String, xtitle: String, ytitle: String, xs: Array[Double], ys: Array[Double]): XYChart = {
    val chart = makeXYChart(title, xtitle, ytitle)
    addLineToChart(chart, None, xs, ys)
  }

  def barChart(
      title: String,
      xtitle: String,
      ytitle: String,
      categories: Seq[String],
      counts: Seq[Int]
  ): CategoryChart = {
    val chart = makeCategoryChart(title, xtitle, ytitle)
    val total = counts.sum
    addBarsToChart(chart, None, categories, counts.map(e => java.lang.Double.valueOf(e * 100.0 / total)))
  }

  def pieChart(title: String, categories: Seq[String], counts: Seq[Int]): PieChart = {
    val chart = makePieChart(title)
    addPieSlicesToChart(chart, categories, counts)
  }

  def histogram(title: String, xtitle: String, ytitle: String, xs: Array[Double], bins: Int): CategoryChart = {
    val chart = makeCategoryChart(title, xtitle, ytitle)
    addHistogramToChart(chart, None, xs, bins)
  }
}
