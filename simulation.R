library(xgboost)
library(dplyr)
library(survival)
library(Hmisc)

set.seed(42)

n <- 1000 

# 1. Simular covariáveis semelhantes às do GBCSG

age <- rnorm(n, mean = 55, sd = 10)
menopausal <- rbinom(n, 1, 0.6)
tumor_size <- rnorm(n, 30, 15)   # em mm
histologic_grade <- sample(1:3, n, replace = TRUE)
positive_nodes <- rpois(n, lambda = 3)
hormone_therapy <- rbinom(n, 1, 0.5)

# 2. Simular tempo verdadeiro via modelo AFT log-normal

beta <- c(age = 0.05, menopausal = 0.4, tumor_size = 0.03, grade2 = 0.2, grade3 = 0.4,
          nodes = 0.2, hormone = 0.3)

# Criar dummies para histologic_grade
grade2 <- as.numeric(histologic_grade == 2)
grade3 <- as.numeric(histologic_grade == 3)

# Linear predictor:

lp <- beta["age"] * age + beta["menopausal"] * menopausal + beta["tumor_size"] * tumor_size +
  beta["grade2"] * grade2 + beta["grade3"] * grade3 +
  beta["nodes"] * positive_nodes + beta["hormone"] * hormone_therapy

summary(lp)
sd(lp)

log_T <- lp + rnorm(n, mean = 0, sd = 1.5)
T_true <- exp(log_T)  #1000 tempos gerados

summary(T_true)

table(T_true>2000)

# 3. Simular número de visitas, agora dependendo do risco (tempo real) 

risk_score <- lp 

num_visits <- round(3 - 2 * (log(T_true) - min(log(T_true))) / (max(log(T_true)) - min(log(T_true))))
num_visits <- pmax(2, pmin(3, num_visits)) 

# 4. Simular tempos de visitas uniformemente distribuídos até um tempo limite (ex: máximo 2x tempo verdadeiro)

L <- R <- rep(NA, n)

for (i in 1:n) {
  #i<-1
  visits <- sort(runif(num_visits[i], 0,  1.7* T_true[i])) 
  
  #ALEATORIO:  1.7: 32.4%,  2.6: 14.7%    3.5: 7.8%    30: 0% 
  
  idx <- which(visits >= T_true[i])[1]   
  
  if (is.na(idx)) {
    # censura à direita
    L[i] <- max(visits)
    R[i] <- NA
  } else if (idx == 1) {
    L[i] <- 0
    R[i] <- visits[1]
  } else {
    L[i] <- visits[idx - 1]
    R[i] <- visits[idx]
  }
}

# Dataset final
data_sim <- data.frame(
  age = age,
  menopausal = menopausal,
  tumor_size = tumor_size,
  histologic_grade = histologic_grade,
  positive_nodes = positive_nodes,
  hormone_therapy = hormone_therapy,
  T_true = T_true,
  L = L,
  R = R
)

data_sim$right_1e9<-ifelse(is.na(data_sim$R), max(data_sim$T_true) , data_sim$R) #1e9   #max(data_sim$T_true)

# Midpoint imputation

library(dplyr)

inputted.data <- data_sim %>%
  mutate(rec.time = ifelse(is.finite(R), (L + R)/2, L),
         status = ifelse(is.finite(R), 1, 0))

# Make data.frame
results <- data.frame(
  true=data_sim$T_true,
  left = data_sim$L,
  right = data_sim$R,
  #predicted = preds,
  #predicted2 = preds2,
  midpoint=inputted.data$rec.time,
  status=inputted.data$status,
  age=inputted.data$age,
  menopausal=inputted.data$menopausal, 
  tumor_size=inputted.data$tumor_size,
  histologic_grade=inputted.data$histologic_grade,
  positive_nodes=inputted.data$positive_nodes,
  hormone_therapy=inputted.data$hormone_therapy,
  rectime=inputted.data$rec.time
)

results.ini<-results

data_sim.ini<-data_sim

# Interval-censored times: model-based imputation + scaled linear redistribution
# Right-censored times (upper bound = Inf): imputed as lower bound to avoid extrapolation

results<-results[!is.na(results$right),]

data_sim<-data_sim[!is.na(data_sim$R),]

#xgboost_scaled:

n_bs <- 1000

xgboost.impt<-NULL

for (k in 1:n_bs) {
  
  cat(k, '\n')
  
  label_lower <- data_sim$L
  label_upper <- data_sim$right_1e9 # == data_sim$R  #Inf
  
  times<-rep(0,nrow(data_sim))
  
  for(i in 1:nrow(data_sim)){
    
    times[i] <- runif(1, min = data_sim$L[i], max = label_upper[i])
  }
  
  
  X <- model.matrix(~ age + menopausal + tumor_size + histologic_grade + 
                      positive_nodes + hormone_therapy , data = data_sim)[, -1] #-1 tira o coeficiente intercept
  
  dtrain <- xgb.DMatrix(data = X)
  
  label_lower <- times
  label_upper <- times
  
  setinfo(dtrain, "label_lower_bound", label_lower)
  setinfo(dtrain, "label_upper_bound", label_upper)
  
  params <- list(
    objective = "survival:aft", #"survival:cox"
    eval_metric = "aft-nloglik",
    aft_loss_distribution = "normal",  #"normal" or "logistic", "extreme"
    aft_loss_distribution_scale = 1.0,
    tree_method = "hist"
  )
  # 5. Treinar o modelo
  bst <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = 100
  )
  # 6. Fazer previsões
  preds3 <- predict(bst, newdata = dtrain)
  xgboost.impt<-cbind(xgboost.impt, preds3)
  
}

medians<-rep(0, nrow(xgboost.impt))

label_lower <- data_sim$L
label_upper <- data_sim$right_1e9

for (j in 1:nrow(xgboost.impt)){
  
  cat(j, '\n')
  
  #j<-1
  
  times_scaled<-rep(0,n_bs)
  
  for(i in 1:n_bs){
    
    times_scaled[i]<-as.numeric(xgboost.impt[j,i])
  }
  
  val<-label_lower[j]+(times_scaled-min(times_scaled))/(max(times_scaled)-min(times_scaled))*(label_upper[j]-label_lower[j])
  
  val>=results$left[j]
  
  val<results$right[j]
  
  medians[j]<-median(val, na.rm = T)
  
}

results$xgb_1e9_scaled<-medians

#logn_scaled:

bs_fits <- vector("list", n_bs)
set.seed(123)  # para reprodutibilidade

for (k in 1:n_bs) {
  
  label_lower==results$left #all true
  results$left==data_sim$L #all true
  
  times<-rep(0,nrow(data_sim))
  
  for(i in 1:nrow(data_sim)){
    
    times[i] <- runif(1, min = data_sim$L[i], max = data_sim$right_1e9[i])
  }
  
  results$times<-times
  
  bs_fits[[k]] <- survreg(formula = Surv(times, status) ~age + menopausal + tumor_size + histologic_grade + 
                            positive_nodes + hormone_therapy,
                          data =  results ,
                          dist="lognormal")
}

medians<-rep(0, nrow(results))

for (j in 1:nrow(results) ){
  
  cat(j, '\n')
  
  #j<-1
  
  times_scaled<-rep(0,n_bs)
  
  for(i in 1:n_bs){
    
    times_scaled[i]<-predict(bs_fits[[i]])[j]
  }
  
  label_lower <- data_sim$L
  label_upper <- data_sim$right_1e9
  
  val<-label_lower[j]+(times_scaled-min(times_scaled))/(max(times_scaled)-min(times_scaled))*(label_upper[j]-label_lower[j])
  
  val>=results$left[j]
  
  val<results$right[j]
  
  medians[j]<-median(val, na.rm = T)

}

results$lnorm_1e9_scaled<-medians

#weibull_scaled:

bs_fits <- vector("list", n_bs)
set.seed(123)  # para reprodutibilidade

for (k in 1:n_bs) {
  
  label_lower==results$left #all true
  results$left==data_sim$L #all true
  
  times<-rep(0,nrow(data_sim))
  
  for(i in 1:nrow(data_sim)){
    
    times[i] <- runif(1, min = data_sim$L[i], max = data_sim$right_1e9[i])
  }
  
  results$times<-times
  
  bs_fits[[k]] <- survreg(formula = Surv(times, status) ~ age + menopausal + tumor_size + histologic_grade + 
                            positive_nodes + hormone_therapy,
                          data =  results ,
                          dist="weibull")
  
}

medians<-rep(0, nrow(results))

for (j in 1:nrow(results) ){
  
  cat(j, '\n')
  
  #j<-1
  
  times_scaled<-rep(0,n_bs)
  
  for(i in 1:n_bs){
    
    times_scaled[i]<-predict(bs_fits[[i]])[j]
  }
  
  label_lower <- data_sim$L
  label_upper <- data_sim$right_1e9
  
  val<-label_lower[j]+(times_scaled-min(times_scaled))/(max(times_scaled)-min(times_scaled))*(label_upper[j]-label_lower[j])
  
  val>=results$left[j]
  
  val<results$right[j]
  
  medians[j]<-median(val, na.rm = T)
  
}

results$wei_1e9_scaled<-medians

#exponential_scaled:

bs_fits <- vector("list", n_bs)
set.seed(123)  # para reprodutibilidade

for (k in 1:n_bs) {
  
  label_lower==results$left #all true
  results$left==data_sim$L #all true
  
  times<-rep(0,nrow(data_sim))
  
  for(i in 1:nrow(data_sim)){
    
    times[i] <- runif(1, min = data_sim$L[i], max = data_sim$right_1e9[i])
  }
  
  results$times<-times
  
  bs_fits[[k]] <- survreg(formula = Surv(times, status) ~ age + menopausal + tumor_size + histologic_grade + 
                            positive_nodes + hormone_therapy,
                          data =  results ,
                          dist="exponential")
}

medians<-rep(0, nrow(results))

for (j in 1:nrow(results) ){
  
  cat(j, '\n')
  
  #j<-1
  
  times_scaled<-rep(0,n_bs)
  
  for(i in 1:n_bs){
    
    times_scaled[i]<-predict(bs_fits[[i]])[j]
  }
  
  label_lower <- data_sim$L
  label_upper <- data_sim$right_1e9
  
  val<-label_lower[j]+(times_scaled-min(times_scaled))/(max(times_scaled)-min(times_scaled))*(label_upper[j]-label_lower[j])
  
  val>=results$left[j]
  
  val<results$right[j]
  
  medians[j]<-median(val, na.rm = T)
  
}

results$exp_1e9_scaled<-medians

#Gaussian_scaled

bs_fits <- vector("list", n_bs)
set.seed(123)  # para reprodutibilidade

for (k in 1:n_bs) {
  
  label_lower==results$left #all true
  results$left==data_sim$L #all true
  
  times<-rep(0,nrow(data_sim))
  
  for(i in 1:nrow(data_sim)){
    
    times[i] <- runif(1, min = data_sim$L[i], max = data_sim$right_1e9[i])
  }
  
  results$times<-times
  
  bs_fits[[k]] <- survreg(formula = Surv(times, status) ~ age + menopausal + tumor_size + histologic_grade + 
                            positive_nodes + hormone_therapy,
                          data =  results ,
                          dist="gaussian")
}

medians<-rep(0, nrow(results))

for (j in 1:nrow(results) ){
  
  cat(j, '\n')
  
  #j<-1
  
  times_scaled<-rep(0,n_bs)
  
  for(i in 1:n_bs){
    
    times_scaled[i]<-predict(bs_fits[[i]])[j]
  }
  
  label_lower <- data_sim$L
  label_upper <- data_sim$right_1e9
  
  val<-label_lower[j]+(times_scaled-min(times_scaled))/(max(times_scaled)-min(times_scaled))*(label_upper[j]-label_lower[j])
  
  val>=results$left[j]
  
  val<results$right[j]
  
  medians[j]<-median(val, na.rm = T)
}

results$gaus_1e9_scaled<-medians

#xgboost_scaled (without applying The Scaled Linear Redistribution Method)

n_bs <- 1

xgboost.impt<-NULL

for (k in 1:n_bs) {
  
  #k<-1
  
  cat(k, '\n')
  
  label_lower <- data_sim$L
  label_upper <- data_sim$right_1e9 # == data_sim$R  #Inf
  
  times<-rep(0,nrow(data_sim))
  
  for(i in 1:nrow(data_sim)){
    
    times[i] <- runif(1, min = data_sim$L[i], max = label_upper[i])
  }
  
  
  X <- model.matrix(~ age + menopausal + tumor_size + histologic_grade + 
                      positive_nodes + hormone_therapy , data = data_sim)[, -1] #-1 tira o coeficiente intercept
  
  
  dtrain <- xgb.DMatrix(data = X)
  
  label_lower <- times
  label_upper <- times
  
  setinfo(dtrain, "label_lower_bound", label_lower)
  setinfo(dtrain, "label_upper_bound", label_upper)
  
  params <- list(
    objective = "survival:aft", #"survival:cox"
    eval_metric = "aft-nloglik",
    aft_loss_distribution = "normal",  #"normal" or "logistic", "extreme"
    aft_loss_distribution_scale = 1.0,
    tree_method = "hist"
  )
    # 5. Treinar o modelo
  bst <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = 100
  )
    # 6. Fazer previsões
  preds3 <- predict(bst, newdata = dtrain)
  xgboost.impt<-cbind(xgboost.impt, preds3)
  
}

results$xgb_1e9_no_scaled<-as.numeric(xgboost.impt)

results.na<-results.ini[is.na(results.ini$right), ]

results.na$xgb_1e9_scaled<-results.na$left     

results.na$lnorm_1e9_scaled<-results.na$left

results.na$wei_1e9_scaled<-results.na$left

results.na$exp_1e9_scaled<-results.na$left 

results.na$gaus_1e9_scaled<-results.na$left

results.na$xgb_1e9_no_scaled<-results.na$left     

results<-results[, -14]

results<-rbind(results.na, results)

#save(list = ls(all.names = TRUE),file="random_results_0per.RData")
#save(list = ls(all.names = TRUE),file="random_results_7.8per.RData")
#save(list = ls(all.names = TRUE),file="random_results_14.7per.RData")
save(list = ls(all.names = TRUE),file="random_results_32.4per.RData")

#load('random_results_0per.RData')  # -- 0%  (censurados)
#load('random_results_7.8per.RData')  # -- 7.8%  (censurados)
#load('random_results_14.7per.RData')  # -- 14.7%  (censurados)
#load('random_results_32.4per.RData')  # -- 32.4%  (censurados)

results<-results[, c('true','left', 'right', 'rectime', 'status', 'midpoint', 
                     'xgb_1e9_scaled', 'lnorm_1e9_scaled', 'wei_1e9_scaled', 
                     'exp_1e9_scaled', 'gaus_1e9_scaled','xgb_1e9_no_scaled')] 

(bias_pm<-round(sum(abs(results$midpoint-results$true)/length(results$true)),3)) 
(bias_xgb_1e9_scaled<-round(sum(abs(results$xgb_1e9_scaled-results$true)/length(results$true)),3)) 
(bias_lnorm_1e9_scaled<-round(sum(abs(results$lnorm_1e9_scaled-results$true)/length(results$true)),3)) 
(bias_wei_1e9_scaled<-round(sum(abs(results$wei_1e9_scaled-results$true)/length(results$true)),3)) 
(bias_exp_1e9_scaled<-round(sum(abs(results$exp_1e9_scaled-results$true)/length(results$true)),3)) 
(bias_gaus_1e9_scaled<-round(sum(abs(results$gaus_1e9_scaled-results$true)/length(results$true)),3)) 
(bias_xgb_1e9_no_scaled<-round(sum(abs(results$xgb_1e9_no_scaled-results$true)/length(results$true)),3)) 

(var_pm<-sum((results$midpoint-results$true)^2)/(length(results$true)-1))
(var_xgb_1e9_scaled<-sum((results$xgb_1e9_scaled-results$true)^2)/(length(results$true)-1))
(var_lnorm_1e9_scaled<-sum((results$lnorm_1e9_scaled-results$true)^2)/(length(results$true)-1))
(var_wei_1e9_scaled<-sum((results$wei_1e9_scaled-results$true)^2)/(length(results$true)-1))
(var_exp_1e9_scaled<-sum((results$exp_1e9_scaled-results$true)^2)/(length(results$true)-1))
(var_gaus_1e9_scaled<-sum((results$gaus_1e9_scaled-results$true)^2)/(length(results$true)-1))
(var_xgb_1e9_no_scaled<-sum((results$xgb_1e9_no_scaled-results$true)^2)/(length(results$true)-1))

(mse_pm<-round(sqrt(bias_pm^2+var_pm),3))
(mse_xgb_1e9_scaled<-round(sqrt(bias_xgb_1e9_scaled^2+var_xgb_1e9_scaled),3))
(mse_lnorm_1e9_scaled<-round(sqrt(bias_lnorm_1e9_scaled^2+var_lnorm_1e9_scaled),3))
(mse_wei_1e9_scaled<-round(sqrt(bias_wei_1e9_scaled^2+var_exp_1e9_scaled),3))
(mse_exp_1e9_scaled<-round(sqrt(bias_exp_1e9_scaled^2+var_exp_1e9_scaled),3))
(mse_gaus_1e9_scaled<-round(sqrt(bias_gaus_1e9_scaled^2+var_gaus_1e9_scaled),3))
(mse_xgb_1e9_no_scaled<-round(sqrt(bias_xgb_1e9_no_scaled^2+var_xgb_1e9_no_scaled),3))

#Median absolute deviation (MAD):

round(median(abs(results$midpoint-results$true)),3)  #[abs(results$midpoint-results$true)!=0])
round(median(abs(results$xgb_1e9_scaled-results$true)),3)
round(median(abs(results$lnorm_1e9_scaled-results$true)),3)
round(median(abs(results$wei_1e9_scaled-results$true)),3)
round(median(abs(results$exp_1e9_scaled-results$true)),3)
round(median(abs(results$gaus_1e9_scaled-results$true)),3)
round(median(abs(results$xgb_1e9_no_scaled-results$true)),3)

# C-index entre imputado e real

rcorr.cens(results$midpoint, Surv(results$true, results$status))[1]
rcorr.cens(results$xgb_1e9_scaled, Surv(results$true, results$status))[1]
rcorr.cens(results$lnorm_1e9_scaled, Surv(results$true, results$status))[1]
rcorr.cens(results$wei_1e9_scaled, Surv(results$true, results$status))[1]
rcorr.cens(results$exp_1e9_scaled, Surv(results$true, results$status))[1]
rcorr.cens(results$gaus_1e9_scaled, Surv(results$true, results$status))[1]
rcorr.cens(results$xgb_1e9_no_scaled, Surv(results$true, results$status))[1]