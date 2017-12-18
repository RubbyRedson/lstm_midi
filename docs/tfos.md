# TensorFlow on Spark
This is a collection of information gathered when converting the basic LSTM-driven TensorFlow implementation found in lstm.py into a robust, distributed implementation that can be launched on [TensorFlow on Spark](TODO).


## Resources
Here's a list of resources I've found useful for getting things up-and-running correctly:
* [Practical RNN and Data Handling](http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/)
* [TF RFF Example](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py)
* [TFOS Repo (with examples)](https://github.com/yahoo/TensorFlowOnSpark)
* [Conversion Guide](https://github.com/yahoo/TensorFlowOnSpark/wiki/Conversion-Guide)
* [TFOS Overview](http://yahoohadoop.tumblr.com/post/157196317141/open-sourcing-tensorflowonspark-distributed-deep)
* [TFOS Documentation](https://yahoo.github.io/TensorFlowOnSpark/)
*

## Conversion
See the [Conversion Guide](https://github.com/yahoo/TensorFlowOnSpark/wiki/Conversion-Guide) for information on how to translate an existing TF program into a TFOS program.

## Data Handling


### Location
It _appears_ that the data is expected to reside on HDFS, implying that we may need a preprocessor to upload it there. I don't quite remember how datasets are stored on Hops, and it may be that it already comes with built-in HopsFS support for datasets, but it's worth to keep in mind.

### Queues and Readers
Loading up data appears a bit more complicated than the simple TFLauncher jobs. In particular, we must create worker queues and queue readers around the input data. This is due to having a chief and multiple workers, forcing us to construct worker queues based  on the input directories (on HDFS, I believe).

Below is a CSV-based example, which resmebles our code the closest. There also appears to be a TFRecord file format (.tfr), which the MNIST example uses to split the input data into multiple queues, piping sections of the input directory to the multiple workers.

**CSV-based MNIST Example:**
```python
def read_csv_data(image_dir, label_dir, batch_size=100, num_epochs=None, task_index=None, num_workers=None):
        # Setup queue of csv image filenames
        tf_record_pattern = os.path.join(image_dir, 'part-*')
        images = tf.gfile.Glob(tf_record_pattern)
        image_queue = tf.train.string_input_producer(
                images,
                shuffle=False,
                capacity=1000,
                num_epochs=num_epochs,
                name="image_queue")

        # Setup queue of csv label filenames
        tf_record_pattern = os.path.join(label_dir, 'part-*')
        labels = tf.gfile.Glob(tf_record_pattern)
        label_queue = tf.train.string_input_producer(
                labels,
                shuffle=False,
                capacity=1000,
                num_epochs=num_epochs,
                name="label_queue")

        # Setup reader for image queue
        img_reader = tf.TextLineReader(name="img_reader")
        _, img_csv = img_reader.read(image_queue)
        image_defaults = [ [1.0] for col in range(784) ]
        img = tf.stack(tf.decode_csv(img_csv, image_defaults))

        # Normalize values to [0,1]
        norm = tf.constant(255, dtype=tf.float32, shape=(784,))
        image = tf.div(img, norm)

        # Setup reader for label queue
        label_reader = tf.TextLineReader(name="label_reader")
        _, label_csv = label_reader.read(label_queue)
        label_defaults = [ [1.0] for col in range(10) ]
        label = tf.stack(tf.decode_csv(label_csv, label_defaults))

        # Return a batch of examples
        return tf.train.batch(
                [image,label],
                batch_size,
                num_threads=args.readers,
                name="batch_csv")


```


## General Notes
* Seems to be similar to launching TF jobs "through" spark, but with more wrapping abstractions
    * Wrapper function taking arguments and a Spark context
    * Need to specify entire TF program in scope of wrapping function, including dependencies and helper functions
    * Core TF code appears to be exactly the same.

* Differences notes
    * Loading up data appears a bit more complicated than the simple TFLauncher jobs
        * This is due to having a chief and multiple workers, forcing us to construct worker queues based  on the input directories (on HDFS, I believe)
        ```python
        images = tf.gfile.Glob(tf_record_pattern)
        image_queue = tf.train.string_input_producer(
            images,
            shuffle=False,
            capacity=1000,
            num_epochs=num_epochs,
            name="image_queue")
        ```



# (GitHub-Flavored) Markdown Editor

Basic useful feature list:

 * Ctrl+S / Cmd+S to save the file
 * Ctrl+Shift+S / Cmd+Shift+S to choose to save as Markdown or HTML
 * Drag and drop a file into here to load it
 * File contents are saved in the URL so you can share files


I'm no good at writing sample / filler text, so go write something yourself.

Look, a list!

 * foo
 * bar
 * baz

And here's some code! :+1:

```javascript
$(function(){
  $('div').html('I am a div.');
});
```

This is [on GitHub](https://github.com/jbt/markdown-editor) so let me know if I've b0rked it somewhere.


Props to Mr. Doob and his [code editor](http://mrdoob.com/projects/code-editor/), from which
the inspiration to this, and some handy implementation hints, came.

### Stuff used to make this:

 * [markdown-it](https://github.com/markdown-it/markdown-it) for Markdown parsing
 * [CodeMirror](http://codemirror.net/) for the awesome syntax-highlighted editor
 * [highlight.js](http://softwaremaniacs.org/soft/highlight/en/) for syntax highlighting in output code blocks
 * [js-deflate](https://github.com/dankogai/js-deflate) for gzipping of data to make it fit in URLs
