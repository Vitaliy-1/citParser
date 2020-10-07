CIT_MAX_SEQUENCE_LENGTH = 150;
CIT_UNKNOWN_LABEL = 'unk';
CIT_PAD_LABEL = 'pad';

class Prediction {
	static PATH_TO_VOCAB = 'model4_js/vocab/word2idx.json';
	static PATH_TO_MODEL = 'model4_js/model.json';
	static PATH_TO_LABELS = 'model4_js/vocab/idx2tag.json';
	
	constructor(word2idx, idx2tag, loadedModel) {
		this.word2idx = word2idx;
		this.idx2tag = idx2tag;
		this.loadedModel = loadedModel;
	}
	
	static builder() {
		return this.load().then(value => {
			let {word2idx, idx2tag, loadedModel} = value;
			return new Prediction(word2idx, idx2tag, loadedModel);
		});
	}
	
	static getRequest(path) {
		return new Promise(function (resolve, reject) {
			let oReq = new XMLHttpRequest();
			oReq.open("GET", path);
			let parsedJSON;
			oReq.onload = function () {
				parsedJSON = JSON.parse(this.responseText);
				resolve(parsedJSON);
			};
			oReq.send();
		});
	}
	
	static getModel(path) {
		return new Promise(function (resolve, reject) {
			resolve(tf.loadLayersModel(path));
		});
	}
	
	static async load() {
		const word2idx = await this.getRequest(this.PATH_TO_VOCAB);
		const idx2tag = await this.getRequest(this.PATH_TO_LABELS);
		const loadedModel = await this.getModel(this.PATH_TO_MODEL);
		return {word2idx, idx2tag, loadedModel};
	}
	
	predict(data, word2idx, idx2tag, loadedModel) {
		let tokenized = [];
		let vectorized = [];
		
		data.forEach(function (reference) {
			let withoutPunctation = reference.replace(/[\u2000-\u206F\u2E00-\u2E7F\\'!"#$%&()*+,\-.\/:;<=>?@\[\]^_`{|}~]/g, " ");
			let wordArray = withoutPunctation.split(/\s+/);
			wordArray = wordArray.filter(function (el) {
				return el !== '';
			});
			
			let tokenizedSentence = [];
			let vectorizedSentence = [];
			// Convert words to integers
			wordArray.forEach(function (word) {
				tokenizedSentence.push(word);
				word = word.toLowerCase();
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
			
			tokenized.push(tokenizedSentence);
			vectorized.push(vectorizedSentence);
		});
		
		let sentenceLabelsPairs = new Map();
		vectorized.forEach(function (sentence, index) {
			// Make prediction and assign label IDs
			let tensor = tf.tensor1d(sentence, 'int32').expandDims(0);
			loadedModel.predict(tensor).argMax(-1).data().then(labelIds => {
				// Create array with predicted labels
				let labelTags = [];
				const correspondentTokenizedSentence = tokenized[index];
				labelIds.forEach(function (id, idIndex) {
					if (idIndex > correspondentTokenizedSentence.length - 1) return;
					labelTags.push(idx2tag[id]);
				});
				// Assign as word array - label tag array pair
				sentenceLabelsPairs.set(correspondentTokenizedSentence, labelTags);
			});
		});
		
		return sentenceLabelsPairs;
	}
}

let prediction = Prediction.builder();

async function makePrediction() {
	let currentPrediction = await prediction;
	
	let citations = document.getElementById('citations').value;
	if (!citations) return;
	const data = citations.split(/\r\n|\r|\n/g);
	return currentPrediction.predict(data, currentPrediction.word2idx, currentPrediction.idx2tag, currentPrediction.loadedModel);
}

function showPrediction() {
	makePrediction().then(predicted => {
		let oldWrapper = document.getElementById('resultsWrapper');
		if (oldWrapper) {
			oldWrapper.remove();
		}
		
		let resultsWrapperEl = document.createElement('main');
		resultsWrapperEl.setAttribute('id', 'resultsWrapper');
		document.body.appendChild(resultsWrapperEl);
		
		// TODO predicted should be defined without addition timeout, wondering what I'm missing in working with asynchronous functions
		setTimeout(function () {
			for (let [words, labels] of predicted.entries()) {
				let tableEl = document.createElement('table');
				resultsWrapperEl.appendChild(tableEl);
				let labelsRowEl = document.createElement('tr')
				let wordsRowEl = document.createElement('tr');
				tableEl.appendChild(labelsRowEl);
				tableEl.appendChild(wordsRowEl);
				labels.forEach(function (value) {
					let labelCellEl = document.createElement('td');
					labelCellEl.append(value);
					labelsRowEl.appendChild(labelCellEl);
				});
				words.forEach(function (value) {
					let wordCellEl = document.createElement('td');
					wordCellEl.append(value);
					wordsRowEl.appendChild(wordCellEl);
				});
			}
		}, 1000);
	})
}
