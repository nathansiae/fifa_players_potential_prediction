# Fifa_Players_Potential_Prediction
<div style="text-align: left;">
<table style="width:100%; background-color:transparent;">
  <tr style="background-color:transparent;">
    <td style="background-color:transparent;"><a href="http://www.datascience-paris-saclay.fr">
<img border="0" src="http://project.inria.fr/saclaycds/files/2017/02/logoUPSayPlusCDS_990.png" width="90%"> </td>
     
  </tr>
</table> 
</div>

**Student Proposed Data Challenge in Data Camp Course of Master Data Science at University Paris-Saclay**

**Students: Jiabin CHEN, Qin WANG, Chuanyuan QIAN**

The challenge is for football⚽️⚽️⚽️ enthusiasts who love data science and FIFA game🎮🎮🎮. 

*Why we choose this dataset as a data challenge ?* One of the most important tasks for a football club manager is to discover talent players, especially young players with high career potential. We want to build a regression model to predict the player's potential based on his statistical data in FIFA20 Game. Insights and correlations among players' potential, age and skills rating can be derived from the dataset. Indeed, even though some of features values are virtual and given by experts based on performance in reality over past few years, we can extend easily our regression model once we have collected some realistic data, such as the number of goals, assists, win games, etc. We have derived summary statistics and provided a very simple baseline in this project.

![FIFA20](https://www.fifplay.com/img/public/fifa-20-logo.png)

#### Set up

Open a terminal and

1. install the `ramp-workflow` library (if not already done)
  ```
  $ pip install git+https://github.com/paris-saclay-cds/ramp-workflow.git
  ```

2. Follow the ramp-kits instructions from the [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki/Getting-started-with-a-ramp-kit)

#### Local notebook

Get started on this RAMP with the [notebook](starting_kit.html).

To test the starting-kit, run


```
ramp_test_submission --submission starting_kit
```
