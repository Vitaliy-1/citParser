(async () => {
	const loadedModel = await tf.loadLayersModel('http://localhost:63342/citParser/model4_js/model.json');
	
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
		wordArray.forEach(function (word) {
		
		})
		
	});
	
})();
