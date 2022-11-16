const video = document.getElementById('video')

Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
    faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
    faceapi.nets.ssdMobilenetv1.loadFromUri('/models')
]).then(startVideo)


function startVideo() {
    navigator.mediaDevices.getUserMedia(
        {video: {}}) .then((stream)=> {video.srcObject = stream;}, (err)=> console.error(err));
}

recognizeFaces()


async function recognizeFaces() {
  const labeledDescriptors = await loadLabeledImages()
  console.log(labeledDescriptors)
  const faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.6);

  video.addEventListener('play', async () => {
      const canvas = faceapi.createCanvasFromMedia(video)
      document.body.append(canvas)
      
      const displaySize = { width: video.width, height: video.height }
      faceapi.matchDimensions(canvas, displaySize)
      
     
      setInterval(async () => {
             
          const detections = await faceapi.detectAllFaces(video).withFaceLandmarks().withFaceDescriptors()
      
          const resizedDetections = faceapi.resizeResults(detections, displaySize)

          canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)

          const results = resizedDetections.map((d) => { 
            return faceMatcher.findBestMatch(d.descriptor)
          })
          results.forEach((results, i) => {
              const box = resizedDetections[i].detection.box
              const drawBox = new faceapi.draw.DrawBox(box, {label: results.toString() })
              drawBox.draw(canvas)
          })        
      }, 100)

  })
}

function loadLabeledImages() {

  const labels = ['mask', 'no_Mask']
  return Promise.all(
    labels.map(async label => {
      const descriptions = []
      for (let i = 1; i <= 3; i++) {
        const img = await faceapi.fetchImage(`../images/${label}/${i}.jpeg`)
        const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()
        descriptions.push(detections.descriptor)
      }

      return new faceapi.LabeledFaceDescriptors(label, descriptions)
    })
  )
}



