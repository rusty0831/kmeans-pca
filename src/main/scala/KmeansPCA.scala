/**
 * Created by rusty_lai on 2015/5/29.
 */
import _root_.org.apache.spark.SparkConf
import _root_.org.apache.spark.SparkContext
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.linalg.distributed.RowMatrix


object KmeansPCA {
  val conf = new SparkConf().setAppName("SVM Test")
  val sc = new SparkContext(conf)

  val rawData = sc.textFile("/root/kddcup.data")
  rawData.map(_.split(',').last).countByValue().toSeq.sortBy(_._2).reverse.foreach(println)

  val labelsAndData = rawData.map { line =>
    val buffer = line.split(',').toBuffer
    buffer.remove(1, 3)
    val label = buffer.remove(buffer.length-1)
    val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
    (label,vector)
  }

  val parsedData = rawData.map { line =>
    val buffer = line.split(',').toBuffer
    buffer.remove(1, 3)
    val label = buffer.remove(buffer.length-1)
    val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
    (vector)
  }

  val data = labelsAndData.values.cache()

  val mat = new RowMatrix(parsedData)
  val pc = mat.computePrincipalComponents(2)

  val projected = mat.multiply(pc).rows

  val kmeans = new KMeans()
  val model = kmeans.run(projected)

  model.clusterCenters.foreach(println)

  println("Cost = " + model.computeCost(projected))
}
