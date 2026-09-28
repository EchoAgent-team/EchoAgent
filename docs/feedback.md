```
(base) ➜  EchoAgent git:(main) ✗ curl -v --max-time 120 http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"prompt": "late-night rainy city drive, no metal music"}'
* Host localhost:8000 was resolved.
* IPv6: ::1
* IPv4: 127.0.0.1
*   Trying [::1]:8000...
* connect to ::1 port 8000 from ::1 port 62870 failed: Connection refused
*   Trying 127.0.0.1:8000...
* Connected to localhost (127.0.0.1) port 8000
> POST /recommend HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/8.7.1
> Accept: */*
> Content-Type: application/json
> Content-Length: 57
> 
* upload completely sent off: 57 bytes
```

Takes really long after this point. Need to figure out the cause for low latency? To check the cause behind this.


```
* upload completely sent off: 57 bytes
< HTTP/1.1 200 OK
< date: Mon, 21 Sep 2026 20:03:18 GMT
< server: uvicorn
< content-length: 5244
< content-type: application/json
< 
{"prompt":"late-night rainy city drive, no metal music","playlist":[{"track_id":"TRARRZU128F4253CA2","title":"b'Je Sais Que La Terre Est Plate'","artist_name":"b'Rapha\\xc3\\xabl'","album_title":"b'Je Sais Que La Terre Est Plate (Deluxe)'","year":null,"genre":"ontheroad","tags":["['on the road', '100']","['Titletracks', '100']"],"score":0.1,"sources":["relational"]},{"track_id":"TRARRJL128F92DED0E","title":"b'On Efface'","artist_name":"b'Julie Zenatti'","album_title":"b'Comme Vous'","year":null,"genre":"french","tags":["['french', '100']","['menu du jour', '25']","['bright voice', '25']"],"score":0.1,"sources":["relational"]},{"track_id":"TRARRUZ128F9307C57","title":"b'Howells Delight'","artist_name":"b'The Baltimore Consort'","album_title":"b'Watkins Ale -  Music of the English Renaissance'","year":null,"genre":null,"tags":[],"score":0.1,"sources":["relational"]},{"track_id":"TRARRWA128F42A0195","title":"b'Martha Served'","artist_name":"b'I Hate Sally'","album_title":"b\"Don't Worry Lady\"","year":null,"genre":"hardcore","tags":["['post-hardcore', '100']","['hardcore-punk', '100']"],"score":0.1,"sources":["relational"]},{"track_id":"TRARRPG12903CD1DE9","title":"b'Zip-A-Dee-Doo-Dah (Song of the South)'","artist_name":"b'Orlando Pops Orchestra'","album_title":"b'Easy Listening: Cartoon Songs'","year":null,"genre":null,"tags":[],"score":0.1,"sources":["relational"]},{"track_id":"TRARRER128F9328521","title":"b'Liquid Time (composition by John Goodsall)'","artist_name":"b'Brand X'","album_title":"b'X Communication : Trilogy II'","year":null,"genre":"chill","tags":["['chill', '100']","['Fusion', '100']"],"score":0.1,"sources":["relational"]},{"track_id":"TRARRYC128F428CCDA","title":"b'Misery Path (From the Privilege of Evil)'","artist_name":"b'Amorphis'","album_title":"b'Karelian Isthmus'","year":null,"genre":null,"tags":[],"score":0.1,"sources":["relational"]},{"track_id":"TRARROY128F42281F7","title":"b'Nuovi Re pt. I I (feat. Tek money - Lady Tambler)'","artist_name":"b'Inoki'","album_title":"b'Nobilt\\xc3\\xa0 di strada'","year":null,"genre":null,"tags":[],"score":0.1,"sources":["relational"]},{"track_id":"TRARREF128F422FD96","title":"b'Halloween'","artist_name":"b'Dead Kennedys'","album_title":"b'Milking The Sacred Cow'","year":null,"genre":"rock","tags":["['punk', '100']","['halloween', '52']","['punk rock', '40']","['hardcore punk', '36']","['hardcore', '28']","['80s', '16']","['jello biafra', '16']","['Old School Punk', '12']","['conservative', '8']","['California', '8']"],"score":0.1,"sources":["relational"]},{"track_id":"TRARRVB128F92F47CA","title":"b'Parto em terras distantes'","artist_name":"b'Brigada Victor Jara'","album_title":"b'Novas Vos Trago'","year":null,"genre":null,"tags":[],"score":0.1,"sources":["relational"]},{"track_id":"TRARRQO128F427B5F5","title":"b'You Eclipsed By Me (Album Version)'","artist_name":"b'Atreyu'","album_title":"b'The Curse'","year":null,"genre":"hardcore","tags":["['metalcore', '100']","['ok biisi', '50']"],"score":0.1,"sources":["relational"]},{"track_id":"TRARRMK12903CDF793","title":"b'Shovel'","artist_name":"b'Mistress'","album_title":"b'In Disgust We Trust'","year":null,"genre":"freejazz","tags":["['Sludge', '100']","['doom metal', '100']","['grindcore', '100']","['sludgecore', '100']","['sludge metal', '100']","['77davez-all-tracks', '100']"],"score":0.1,"sources":["relational"]},{"track_id":"TRARUOP12903CF2384","title":"b'What Drives The Weak'","artist_name":"b'Shadows Fall'","album_title":"b'The War Within'","year":null,"genre":"hardcore","tags":["['metalcore', '100']","['metal', '81']","['thrash metal', '62']","['heavy metal', '28']","['Shadows Fall', '28']","['rock', '9']","['Melodic Death Metal', '9']","['4-STAR', '6']","['nice solo', '6']","['greatest songs ever', '6']"],"score":0.1,"sources":["relational"]},{"track_id":"TRARURM128F931A91B","title":"b'Life Force'","artist_name":"b'Vanessa Daou'","album_title":"b'Joe Sent Me'","year":null,"genre":null,"tags":[],"score":0.1,"sources":["relational"]},{"track_id":"TRARUDQ128F934B0ED","title":"b'The Dance Of Europe'","artist_name* Connection #0 to host localhost left intact
":"b'Dave Brockie Experience'","album_title":"b'Diarrhea Of A Madman'","year":null,"genre":null,"tags":[],"score":0.1,"sources":["relational"]}],"debug":{"intent":{"semantic_query":"late night rainy city drive mood:moody scene:night_city energy:low","hard_constraints":{},"soft_preferences":{},"exclusions":{"genres_exclude":["metal"]}},"plan":{"playlist_size":15,"semantic_weight":0.65,"relational_weight":0.1,"soft_preference_weight":0.15,"novelty_weight":0.1,"artist_repeat_penalty":0.3,"genre_concentration_penalty":0.2,"exclusion_penalty":0.5,"retrieval_limits":{"n_vector":150,"n_relational":100},"broaden_if_low_recall":true,"diversity_strictness":"low","rationale":"The request is driven by a specific late‑night rainy mood, so we give the highest weight to semantic similarity, keep relational and soft‑preference weights low, and use only a modest novelty boost while penalizing metal exclusions and artist repeats."},"relational_candidate_count":100,"vector_candidate_count":0,"fused_candidate_count":0,"retry_count":1,"critic_report":{"accept":true,"reason":"no critic_agent in state — auto-accepted","suggested_adjustments":{}}}}%      
```

This is the output generated after a while. Following issues noted:
- playlist got accepted despite issues since critic agent is missing
- noticed it is missing originally from playlist_graph, even though we have tested and it works in notebooks.
- why is exclusion_penalty only 0.5? Need to revisit scoring
- Time limit and error message - no clean exit it seems when json is not generated properly, and max tries is exhausted
- How does the exclusion work? Is it only excluding seed genre and not looking at the top tags? Also maybe hard exlcusions or inclusions cannot simply be a relational discard or addition, needs semantic similarity search, therefore also a RAG thing and/or agent that decides? Can critic agent alone handle it? Second order problem. To be deferred to the future.
- 