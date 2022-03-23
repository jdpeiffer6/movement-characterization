## To Do:
### Ideas
- [ ] Quantify differences in variances for most measures.
- [ ] Try shapelet transform or unsupervised learning task on pelvis velocity profiles.
### Analysis
- [x] Normalize jerk to dimensionless term.
- [x] Look at tandem analysis.
- [ ] Analyze Shank Movements
- [ ] Analyze 9-Hole Peg Test
### Experiments
- [x] Test new 9HP protocol.

## Notes:

Meeting 3/22/22
- The pelvis velocity profiles between the control and subject with cervical myelopathy are very different. We thought this could be a good unsupervised learning task in the future.
- Brenton mentions that the finer metrics I have been devoloping here will possibly be most useful for young people who are able to compesate well for balance issues.
- The variance of many measues (like hip angle, pelvis jerk, etc) is usally much larger for the subject with CM. So instead of only looking at the difference in means, we could look at the difference in variances.


Meeting 3/3/22
- Brenton mentions that we maybe should not me normalizing by velocity...he is in fact correct! We should be normalizing by D^5/A^2, where D is duration of movement and A is movement amplitude. Because V_{mean} = A/D, you can also normalize to D^3/v^2_{mean}...so what I was doing was close to being right...I just needed to think about it a bit more! See Hogan and Sternad 2009 and von Kordelaar 2014.
- Also mentioned that we should quantify the straightness of walking. Even if is straight, we need to quantify if it deviates at an angle.
- I also mentioned we could add a shoulder jerk measure. Time will tell if this is complimentary to thorax or not.


