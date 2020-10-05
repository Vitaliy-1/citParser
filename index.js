const CIT_MAX_SEQUENCE_LENGTH = 150;
const CIT_UNKNOWN_LABEL = 'unk';
const CIT_PAD_LABEL = 'pad';

const pathToVocab = 'model4_js/vocab/word2idx.json';
const pathToModel = 'http://localhost:63342/citParser/model4_js/model.json';

function getRequest(path) {
	return new Promise(function (resolve, reject) {
		var oReq = new XMLHttpRequest();
		oReq.open("GET", path);
		let parsedJSON;
		oReq.onload = function () {
			parsedJSON = JSON.parse(this.responseText);
			resolve(parsedJSON);
		};
		oReq.send();
	});
}

function getModel(path) {
	return new Promise(function (resolve, reject) {
		 resolve(tf.loadLayersModel(path))
	});
}

async function predict() {
	const word2idx = await getRequest(pathToVocab);
	const loadedModel = await getModel(pathToModel);
	
	var newData = [
		'Baile W, Buckman R, Lenzi R, Glober G, Beale E, Kudelka A. SPIKES-A Six-Step Protocol for Delivering Bad News: Application to the Patient with Cancer. The Oncologist. 2000;5(4):302-11.',
		'Patel K, Tatham K. Complete OSCE Skills for Medical and Surgical Finals. London: Hodder Arnold; 2010.',
		'Burton N, Birdi K. Clinical Skills for OSCEs. London: Informa; 2006.'];
	
	var vectorized = [];
	
	newData.forEach(function (reference) {
		var withoutPunctation = reference.replace(/[\u2000-\u206F\u2E00-\u2E7F\\'!"#$%&()*+,\-.\/:;<=>?@\[\]^_`{|}~]/g, " ");
		var wordArray = withoutPunctation.split(/\s+/);
		wordArray = wordArray.filter(function (el) {
			return el !== '';
		});
		
		var vectorizedSentence = [];
		// Convert words to integers
		wordArray.forEach(function (word) {
			if (word2idx.hasOwnProperty(word)) {
				vectorizedSentence.push(word2idx[word]);
			} else {
				vectorizedSentence.push(word2idx[CIT_UNKNOWN_LABEL])
			}
		});
		
		// Fill with zeros
		const sentenceLength = vectorizedSentence.length
		if (sentenceLength < CIT_MAX_SEQUENCE_LENGTH) {
			const arrayFill = new Array(CIT_MAX_SEQUENCE_LENGTH - sentenceLength).fill(word2idx[CIT_PAD_LABEL]);
			arrayFill.forEach(function (value) {
				vectorizedSentence.push(value)
			});
		}

		vectorized.push(vectorizedSentence);
	});
	
	predictedLabels = [];
	vectorized.forEach(function (sentence) {
		let tensor = tf.tensor1d(sentence, dtype='int32').expandDims(0);
		const predicted = loadedModel.predict(tensor).argMax(-1).data();
		predictedLabels.push(predicted)
	})
}

const predicted = predict();
