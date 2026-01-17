library(interval)
library(xgboost)
library(survival)
library(icenReg)
library(dplyr)

data("IR_diabetes")

db<-IR_diabetes

db$left_rec<-db$left

db$right_rec<-db$right

db$right_1e9<-ifelse(is.na(db$right_rec), 1e9, db$right_rec)

# Midpoint imputation

library(dplyr)

inputted.data <- db %>%
  mutate(rec.time = ifelse(is.finite(right_rec), (left_rec + right_rec)/2, left_rec),
         status = ifelse(is.finite(right_rec), 1, 0))

head(inputted.data)  #"rec.time", "status" --- two news column -- midpoint

# Make data.frame
results <- data.frame(
  left = db$left,
  right = db$right,
  right_rec = db$right_rec,
  midpoint=inputted.data$rec.time,
  status=inputted.data$status,
  gender=inputted.data$gender,
  right_1e9=db$right_1e9
)

head(results)


db.ini<-db

results.ini<-results

#xgboost_scaled

db$gender<-as.numeric(db$gender)

n_bs <- 1000

xgboost.impt<-NULL

for (k in 1:n_bs) {
  
  cat(k, '\n')
  
  label_lower <- db$left
  label_upper <- results$right_1e9
  
  times<-rep(0,nrow(results)) 
  
  for(i in 1:nrow(results)){
    
    times[i] <- runif(1, min = db$left[i], max = label_upper[i])
  }
  
  X <- model.matrix(~ gender, data = results)#[, -1] #-1 tira o coeficiente intercept (retirei estava a dar erro)
  
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

label_lower <- results$left
label_upper <- results$right_1e9

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


#logn_scaled

bs_fits <- vector("list", n_bs)
set.seed(123)  # para reprodutibilidade

for (k in 1:n_bs) {
  
  times<-rep(0,nrow(results))
  
  for(i in 1:nrow(results)){
    
    times[i] <- runif(1, min = db$left[i], max = results$right_1e9[i])
  }
  
  results$times<-times
  
  bs_fits[[k]] <- survreg(formula = Surv(times, status) ~gender,
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
  
  label_lower <- db$left
  label_upper <- results$right_1e9
  
  val<-label_lower[j]+(times_scaled-min(times_scaled))/(max(times_scaled)-min(times_scaled))*(label_upper[j]-label_lower[j])
  
  val>=results$left[j]
  
  val<results$right[j]
  
  medians[j]<-median(val, na.rm = T)
  
  
}

results$lnorm_1e9_scaled<-medians

#WEIBULL_scaled

bs_fits <- vector("list", n_bs)
set.seed(123)  # para reprodutibilidade

for (k in 1:n_bs) {
  
  times<-rep(0,nrow(results))
  
  for(i in 1:nrow(results)){
    
    times[i] <- runif(1, min = db$left[i], max = results$right_1e9[i])
  }
  
  results$times<-times
  
  bs_fits[[k]] <- survreg(formula = Surv(times, status) ~  gender,
                          data =  results,
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
  
  label_lower <- db$left
  label_upper <- results$right_1e9
  
  val<-label_lower[j]+(times_scaled-min(times_scaled))/(max(times_scaled)-min(times_scaled))*(label_upper[j]-label_lower[j])
  
  val>=results$left[j]
  
  val<results$right[j]
  
  medians[j]<-median(val, na.rm = T)
  
}

results$wei_1e9_scaled<-medians


#exponential_scaled

bs_fits <- vector("list", n_bs)
set.seed(123)  # para reprodutibilidade

for (k in 1:n_bs) {
  
  times<-rep(0,nrow(results))
  
  for(i in 1:nrow(results)){
    
    times[i] <- runif(1, min = db$left[i], max = results$right_1e9[i])
  }
  
  results$times<-times
  
  bs_fits[[k]] <- survreg(formula = Surv(times, status) ~  gender,
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
  
  label_lower <- db$left
  label_upper <- results$right_1e9
  
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
  
  times<-rep(0,nrow(results))
  
  for(i in 1:nrow(results)){
    
    times[i] <- runif(1, min = db$left[i], max = results$right_1e9[i])
  }
  
  results$times<-times
  
  bs_fits[[k]] <- survreg(formula = Surv(times, status) ~  gender,
                          data =  results,
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
  
  label_lower <- db$left
  label_upper <- results$right_1e9
  
  val<-label_lower[j]+(times_scaled-min(times_scaled))/(max(times_scaled)-min(times_scaled))*(label_upper[j]-label_lower[j])
  
  val>=results$left[j]
  
  val<results$right[j]
  
  medians[j]<-median(val, na.rm = T)
  
  
}

results$gaus_1e9_scaled<-medians

results$xgb_1e9_scaled<-ifelse(results$right=='Inf',results$left , results$xgb_1e9_scaled)

results$lnorm_1e9_scaled<-ifelse(results$right=='Inf',results$left , results$lnorm_1e9_scaled)

results$wei_1e9_scaled<-ifelse(results$right=='Inf',results$left , results$wei_1e9_scaled)

results$exp_1e9_scaled<-ifelse(results$right=='Inf',results$left , results$exp_1e9_scaled)

results$gaus_1e9_scaled<-ifelse(results$right=='Inf',results$left , results$gaus_1e9_scaled)

names(results)

results$status_t<-rep(1, nrow(results))

leftTB<-results$left

rightTB<-results$right

fit.int<-icfit(Surv(leftTB, rightTB, type="interval2") ~ 1)

fit.pm<- survfit(Surv(midpoint, status) ~ 1, data = results)  #status_t

fit.xgh<- survfit(Surv(xgb_1e9_scaled, status) ~ 1, data = results)

fit.lnorm<- survfit(Surv(lnorm_1e9_scaled, status) ~ 1, data = results)

fit.wei<- survfit(Surv(wei_1e9_scaled, status) ~ 1, data = results)

fit.exp<- survfit(Surv(exp_1e9_scaled, status) ~ 1, data = results)

fit.gaus<- survfit(Surv(gaus_1e9_scaled, status) ~ 1, data = results)

#uniformizar PM
dados <- data.frame(time = fit.pm$time,
                    surv = fit.pm$surv)
# --- 1. Criar sequência de tempos uniformes ---
tempos_uniformes <- seq(0, max(dados$time), by = 0.5)
# --- 2. Para cada tempo da sequência, encontrar o valor de sobrevivência mais próximo ---
library(dplyr)
dados_uniformes <- data.frame(time = tempos_uniformes) %>%
  rowwise() %>%
  mutate(surv = dados$surv[which.min(abs(dados$time - time))]) %>%
  ungroup()
# --- 3. Arredondar e visualizar ---
dados_uniformes <- dados_uniformes %>%
  mutate(time = round(time, 1),
         surv = round(surv, 3))
# --- 4. Mostrar primeiras linhas ---
dados_uniformes_PM<-dados_uniformes
tab_PM<-as.data.frame(dados_uniformes_PM)

#uniformizar xgb

dados <- data.frame(time = fit.xgh$time,
                    surv = fit.xgh$surv)
# --- 1. Criar sequência de tempos uniformes ---
tempos_uniformes <- seq(0, max(fit.pm$time), by = 0.5)

# --- 2. Para cada tempo da sequência, encontrar o valor de sobrevivência mais próximo ---
dados_uniformes <- data.frame(time = tempos_uniformes) %>%
  rowwise() %>%
  mutate(surv = dados$surv[which.min(abs(dados$time - time))]) %>%
  ungroup()
# --- 3. Arredondar e visualizar ---
dados_uniformes <- dados_uniformes %>%
  mutate(time = round(time, 1),
         surv = round(surv, 3))
# --- 4. Mostrar primeiras linhas ---
dados_uniformes_XGBoost<-dados_uniformes
tab_XGBoost<-as.data.frame(dados_uniformes_XGBoost)

#uniformizar lnorm
dados <- data.frame(time = fit.lnorm$time,
                    surv = fit.lnorm$surv)
# --- 1. Criar sequência de tempos uniformes ---
tempos_uniformes <- seq(0, max(dados$time), by = 0.5)
# --- 2. Para cada tempo da sequência, encontrar o valor de sobrevivência mais próximo ---

dados_uniformes <- data.frame(time = tempos_uniformes) %>%
  rowwise() %>%
  mutate(surv = dados$surv[which.min(abs(dados$time - time))]) %>%
  ungroup()
# --- 3. Arredondar e visualizar ---
dados_uniformes <- dados_uniformes %>%
  mutate(time = round(time, 1),
         surv = round(surv, 3))
# --- 4. Mostrar primeiras linhas ---
dados_uniformes_lnorm<-dados_uniformes
tab_lnorm<-as.data.frame(dados_uniformes_lnorm)

#uniformizar WEIBULL
dados <- data.frame(time = fit.wei$time,
                    surv = fit.wei$surv)
# --- 1. Criar sequência de tempos uniformes ---
tempos_uniformes <- seq(0, max(dados$time), by = 0.5)
# --- 2. Para cada tempo da sequência, encontrar o valor de sobrevivência mais próximo ---
dados_uniformes <- data.frame(time = tempos_uniformes) %>%
  rowwise() %>%
  mutate(surv = dados$surv[which.min(abs(dados$time - time))]) %>%
  ungroup()
# --- 3. Arredondar e visualizar ---
dados_uniformes <- dados_uniformes %>%
  mutate(time = round(time, 1),
         surv = round(surv, 3))
# --- 4. Mostrar primeiras linhas ---
dados_uniformes_wei<-dados_uniformes
tab_wei<-as.data.frame(dados_uniformes_wei)

#uniformizar EXP
dados <- data.frame(time = fit.exp$time,
                    surv = fit.exp$surv)
# --- 1. Criar sequência de tempos uniformes ---
tempos_uniformes <- seq(0, max(dados$time), by = 0.5)
# --- 2. Para cada tempo da sequência, encontrar o valor de sobrevivência mais próximo ---
dados_uniformes <- data.frame(time = tempos_uniformes) %>%
  rowwise() %>%
  mutate(surv = dados$surv[which.min(abs(dados$time - time))]) %>%
  ungroup()
# --- 3. Arredondar e visualizar ---
dados_uniformes <- dados_uniformes %>%
  mutate(time = round(time, 1),
         surv = round(surv, 3))
# --- 4. Mostrar primeiras linhas ---
dados_uniformes_exp<-dados_uniformes
tab_exp<-as.data.frame(dados_uniformes_exp)

#uniformizar GAUSS
dados <- data.frame(time = fit.gaus$time,
                    surv = fit.gaus$surv)
# --- 1. Criar sequência de tempos uniformes ---
tempos_uniformes <- seq(0, max(dados$time), by = 0.5)
# --- 2. Para cada tempo da sequência, encontrar o valor de sobrevivência mais próximo ---
dados_uniformes <- data.frame(time = tempos_uniformes) %>%
  rowwise() %>%
  mutate(surv = dados$surv[which.min(abs(dados$time - time))]) %>%
  ungroup()
# --- 3. Arredondar e visualizar ---
dados_uniformes <- dados_uniformes %>%
  mutate(time = round(time, 1),
         surv = round(surv, 3))
# --- 4. Mostrar primeiras linhas ---
dados_uniformes_gaus<-dados_uniformes
tab_gaus<-as.data.frame(dados_uniformes_gaus)

tab_PM<-tab_PM[1:3500,]
tab_XGBoost<-tab_XGBoost[1:3500,]
tab_lnorm<-tab_lnorm[1:3500,]
tab_wei<-tab_wei[1:3500,]
tab_exp<-tab_exp[1:3500,]
tab_gaus<-tab_gaus[1:3500,]

li<-c(fit.int$intmap[1,] )
sur<-c(1-cumsum(fit.int$pf))

cbind(li, sur)

#PLOT (FIG.1)

plot(fit.int, ylim=c(0,1)) #, xlim=c(0,1050)
lines(tab_PM[1:2000,1], tab_PM[1:2000,2],  col=2,lty = 1, lwd = 1.2)
lines(tab_XGBoost[1:2000,1], tab_XGBoost[1:2000,2],  col=3,lty = 1, lwd = 1.2)
lines(tab_lnorm[1:2000,1], tab_lnorm[1:2000,2],  col=4,lty = 2, lwd = 1.2)

lines(tab_wei[1:2800,1], tab_wei[1:2800,2],  col=2,lty = 2, lwd = 1.2)
lines(tab_exp[1:2800,1], tab_exp[1:2800,2],  col=3,lty = 2, lwd = 1.2)
lines(tab_gaus[1:2800,1], tab_gaus[1:2800,2],  col=4,lty = 2, lwd = 1.2)

legend("topright", legend = c("TB", "Midpoint", "XGBoost", "Log-normal", "Weibull",
                              "Exponential", "Gauss"), lwd = 1.2, 
       col = c(1,2,3,4, 2, 3, 4), cex = 0.5, 
       lty = c(1,1,1,1, 2, 2,2))


### BCOS

data("bcos")

db<-bcos


db$left_rec<-db$left

db$right_rec<-ifelse(db$right=='Inf', NA, db$right)

db$right_1e9<-ifelse(is.na(db$right_rec), 1e9, db$right_rec)

# Midpoint imputation
library(dplyr)

inputted.data <- db %>%
  mutate(rec.time = ifelse(is.finite(right_rec), (left_rec + right_rec)/2, left_rec),
         status = ifelse(is.finite(right_rec), 1, 0))

# Make data.frame
results <- data.frame(
  left = db$left,
  right = db$right,
  right_rec = db$right_rec,
  midpoint=inputted.data$rec.time,
  status=inputted.data$status,
  treatment=inputted.data$treatment,
  right_1e9=db$right_1e9
)

# Interval-censored times: model-based imputation + scaled linear redistribution
# Right-censored times (upper bound = Inf): imputed as lower bound to avoid extrapolation

db.ini<-db

db<-db[!is.na(db$right_rec),] 
  
results.ini<-results

results<-results[!is.na(results$right_rec),]

#xgboost_scaled

db$treatment<-as.numeric(db$treatment)

n_bs <- 1000

xgboost.impt<-NULL

for (k in 1:n_bs) {
  
  cat(k, '\n')
  
  label_lower <- db$left
  label_upper <- results$right_1e9
  
  times<-rep(0,nrow(results)) 
  
  for(i in 1:nrow(results)){
    
    times[i] <- runif(1, min = db$left[i], max = label_upper[i])
  }
  
  X <- model.matrix(~ treatment, data = results)#[, -1] #-1 tira o coeficiente intercept (retirei estava a dar erro)
  
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

label_lower <- results$left
label_upper <- results$right_1e9

for (j in 1:nrow(xgboost.impt)){
  
  cat(j, '\n')

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

head(results)


#logn_scaled

bs_fits <- vector("list", n_bs)
set.seed(123)  # para reprodutibilidade

for (k in 1:n_bs) {
  
  times<-rep(0,nrow(results))
  
  for(i in 1:nrow(results)){
    
    times[i] <- runif(1, min = db$left[i], max = results$right_1e9[i])
  }
  
  results$times<-times
  
  bs_fits[[k]] <- survreg(formula = Surv(times, status) ~treatment,
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
  
  label_lower <- db$left
  label_upper <- results$right_1e9
  
  val<-label_lower[j]+(times_scaled-min(times_scaled))/(max(times_scaled)-min(times_scaled))*(label_upper[j]-label_lower[j])
  
  val>=results$left[j]
  
  val<results$right[j]
  
  medians[j]<-median(val, na.rm = T)
  
}

results$lnorm_1e9_scaled<-medians

#WEIBULL_scaled

bs_fits <- vector("list", n_bs)
set.seed(123)  

for (k in 1:n_bs) {
  
  times<-rep(0,nrow(results))
  
  for(i in 1:nrow(results)){
    
    times[i] <- runif(1, min = db$left[i], max = results$right_1e9[i])
  }
  
  results$times<-times
  
  bs_fits[[k]] <- survreg(formula = Surv(times, status) ~  treatment,
                          data =  results,
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
  
  label_lower <- db$left
  label_upper <- results$right_1e9
  
  val<-label_lower[j]+(times_scaled-min(times_scaled))/(max(times_scaled)-min(times_scaled))*(label_upper[j]-label_lower[j])
  
  val>=results$left[j]
  
  val<results$right[j]
  
  medians[j]<-median(val, na.rm = T)
  
  
}

results$wei_1e9_scaled<-medians

#exponential_scaled

bs_fits <- vector("list", n_bs)
set.seed(123) 

for (k in 1:n_bs) {
  
  times<-rep(0,nrow(results))
  
  for(i in 1:nrow(results)){
    
    times[i] <- runif(1, min = db$left[i], max = results$right_1e9[i])
  }
  
  results$times<-times
  
  bs_fits[[k]] <- survreg(formula = Surv(times, status) ~  treatment,
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
  
  label_lower <- db$left
  label_upper <- results$right_1e9
  
  val<-label_lower[j]+(times_scaled-min(times_scaled))/(max(times_scaled)-min(times_scaled))*(label_upper[j]-label_lower[j])
  
  val>=results$left[j]
  
  val<results$right[j]
  
  medians[j]<-median(val, na.rm = T)
  
}

results$exp_1e9_scaled<-medians

#Gaussian_scaled

bs_fits <- vector("list", n_bs)
set.seed(123) 

for (k in 1:n_bs) {
  
  times<-rep(0,nrow(results))
  
  for(i in 1:nrow(results)){
    
    times[i] <- runif(1, min = db$left[i], max = results$right_1e9[i])
  }
  
  results$times<-times
  
  bs_fits[[k]] <- survreg(formula = Surv(times, status) ~  treatment,
                          data =  results,
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
  
  label_lower <- db$left
  label_upper <- results$right_1e9
  
  val<-label_lower[j]+(times_scaled-min(times_scaled))/(max(times_scaled)-min(times_scaled))*(label_upper[j]-label_lower[j])
  
  val>=results$left[j]
  
  val<results$right[j]
  
  medians[j]<-median(val, na.rm = T)
  
  
}

results$gaus_1e9_scaled<-medians

results.na<-results.ini[is.na(results.ini$right_rec), ]

results.na$xgb_1e9_scaled<-results.na$left     

results.na$lnorm_1e9_scaled<-results.na$left

results.na$wei_1e9_scaled<-results.na$left

results.na$exp_1e9_scaled<-results.na$left 

results.na$gaus_1e9_scaled<-results.na$left

names(results.na)

names(results)

results<-results[, -9]

names(results)

results<-rbind(results.na, results)

results

dim(results)

results$status_t<-rep(1, nrow(results))

leftTB<-results$left

rightTB<-results$right

fit.int<-icfit(Surv(leftTB, rightTB, type="interval2") ~ 1)

fit.pm<- survfit(Surv(midpoint, status) ~ 1, data = results)  #status_t

fit.xgh<- survfit(Surv(xgb_1e9_scaled, status) ~ 1, data = results)

fit.lnorm<- survfit(Surv(lnorm_1e9_scaled, status) ~ 1, data = results)

fit.wei<- survfit(Surv(wei_1e9_scaled, status) ~ 1, data = results)

fit.exp<- survfit(Surv(exp_1e9_scaled, status) ~ 1, data = results)

fit.gaus<- survfit(Surv(gaus_1e9_scaled, status) ~ 1, data = results)

#uniformizar PM
dados <- data.frame(time = fit.pm$time,
                    surv = fit.pm$surv)
# --- 1. Criar sequência de tempos uniformes ---
tempos_uniformes <- seq(0, max(dados$time), by = 0.5)
# --- 2. Para cada tempo da sequência, encontrar o valor de sobrevivência mais próximo ---
library(dplyr)
dados_uniformes <- data.frame(time = tempos_uniformes) %>%
  rowwise() %>%
  mutate(surv = dados$surv[which.min(abs(dados$time - time))]) %>%
  ungroup()
# --- 3. Arredondar e visualizar ---
dados_uniformes <- dados_uniformes %>%
  mutate(time = round(time, 1),
         surv = round(surv, 3))
# --- 4. Mostrar primeiras linhas ---
dados_uniformes_PM<-dados_uniformes
tab_PM<-as.data.frame(dados_uniformes_PM)

#uniformizar xgb

dados <- data.frame(time = fit.xgh$time,
                    surv = fit.xgh$surv)
# --- 1. Criar sequência de tempos uniformes ---
tempos_uniformes <- seq(0, max(fit.pm$time), by = 0.5)

# --- 2. Para cada tempo da sequência, encontrar o valor de sobrevivência mais próximo ---
dados_uniformes <- data.frame(time = tempos_uniformes) %>%
  rowwise() %>%
  mutate(surv = dados$surv[which.min(abs(dados$time - time))]) %>%
  ungroup()
# --- 3. Arredondar e visualizar ---
dados_uniformes <- dados_uniformes %>%
  mutate(time = round(time, 1),
         surv = round(surv, 3))
# --- 4. Mostrar primeiras linhas ---
dados_uniformes_XGBoost<-dados_uniformes
tab_XGBoost<-as.data.frame(dados_uniformes_XGBoost)

#uniformizar lnorm
dados <- data.frame(time = fit.lnorm$time,
                    surv = fit.lnorm$surv)
# --- 1. Criar sequência de tempos uniformes ---
tempos_uniformes <- seq(0, max(dados$time), by = 0.5)
# --- 2. Para cada tempo da sequência, encontrar o valor de sobrevivência mais próximo ---

dados_uniformes <- data.frame(time = tempos_uniformes) %>%
  rowwise() %>%
  mutate(surv = dados$surv[which.min(abs(dados$time - time))]) %>%
  ungroup()
# --- 3. Arredondar e visualizar ---
dados_uniformes <- dados_uniformes %>%
  mutate(time = round(time, 1),
         surv = round(surv, 3))
# --- 4. Mostrar primeiras linhas ---
dados_uniformes_lnorm<-dados_uniformes
tab_lnorm<-as.data.frame(dados_uniformes_lnorm)

#uniformizar WEIBULL
dados <- data.frame(time = fit.wei$time,
                    surv = fit.wei$surv)
# --- 1. Criar sequência de tempos uniformes ---
tempos_uniformes <- seq(0, max(dados$time), by = 0.5)
# --- 2. Para cada tempo da sequência, encontrar o valor de sobrevivência mais próximo ---
dados_uniformes <- data.frame(time = tempos_uniformes) %>%
  rowwise() %>%
  mutate(surv = dados$surv[which.min(abs(dados$time - time))]) %>%
  ungroup()
# --- 3. Arredondar e visualizar ---
dados_uniformes <- dados_uniformes %>%
  mutate(time = round(time, 1),
         surv = round(surv, 3))
# --- 4. Mostrar primeiras linhas ---
dados_uniformes_wei<-dados_uniformes
tab_wei<-as.data.frame(dados_uniformes_wei)

#uniformizar EXP
dados <- data.frame(time = fit.exp$time,
                    surv = fit.exp$surv)
# --- 1. Criar sequência de tempos uniformes ---
tempos_uniformes <- seq(0, max(dados$time), by = 0.5)
# --- 2. Para cada tempo da sequência, encontrar o valor de sobrevivência mais próximo ---
dados_uniformes <- data.frame(time = tempos_uniformes) %>%
  rowwise() %>%
  mutate(surv = dados$surv[which.min(abs(dados$time - time))]) %>%
  ungroup()
# --- 3. Arredondar e visualizar ---
dados_uniformes <- dados_uniformes %>%
  mutate(time = round(time, 1),
         surv = round(surv, 3))
# --- 4. Mostrar primeiras linhas ---
dados_uniformes_exp<-dados_uniformes
tab_exp<-as.data.frame(dados_uniformes_exp)

#uniformizar GAUSS
dados <- data.frame(time = fit.gaus$time,
                    surv = fit.gaus$surv)
# --- 1. Criar sequência de tempos uniformes ---
tempos_uniformes <- seq(0, max(dados$time), by = 0.5)
# --- 2. Para cada tempo da sequência, encontrar o valor de sobrevivência mais próximo ---
dados_uniformes <- data.frame(time = tempos_uniformes) %>%
  rowwise() %>%
  mutate(surv = dados$surv[which.min(abs(dados$time - time))]) %>%
  ungroup()
# --- 3. Arredondar e visualizar ---
dados_uniformes <- dados_uniformes %>%
  mutate(time = round(time, 1),
         surv = round(surv, 3))
# --- 4. Mostrar primeiras linhas ---
dados_uniformes_gaus<-dados_uniformes
tab_gaus<-as.data.frame(dados_uniformes_gaus)

tab_PM<-tab_PM[1:3500,]
tab_XGBoost<-tab_XGBoost[1:3500,]
tab_lnorm<-tab_lnorm[1:3500,]
tab_wei<-tab_wei[1:3500,]
tab_exp<-tab_exp[1:3500,]
tab_gaus<-tab_gaus[1:3500,]

li<-c(fit.int$intmap[1,] )
sur<-c(1-cumsum(fit.int$pf))

cbind(li, sur)

#PLOT (FIG.1)

plot(fit.int, ylim=c(0,1)) #, xlim=c(0,1050)
lines(tab_PM[1:2000,1], tab_PM[1:2000,2],  col=2,lty = 1, lwd = 1.2)
lines(tab_XGBoost[1:2000,1], tab_XGBoost[1:2000,2],  col=3,lty = 1, lwd = 1.2)
lines(tab_lnorm[1:2000,1], tab_lnorm[1:2000,2],  col=4,lty = 2, lwd = 1.2)

lines(tab_wei[1:2800,1], tab_wei[1:2800,2],  col=2,lty = 2, lwd = 1.2)
lines(tab_exp[1:2800,1], tab_exp[1:2800,2],  col=3,lty = 2, lwd = 1.2)
lines(tab_gaus[1:2800,1], tab_gaus[1:2800,2],  col=4,lty = 2, lwd = 1.2)

legend("topright", legend = c("TB", "Midpoint", "XGBoost", "Log-normal", "Weibull",
                              "Exponential", "Gauss"), lwd = 1.2, 
       col = c(1,2,3,4, 2, 3, 4), cex = 0.5, 
       lty = c(1,1,1,1, 2, 2,2))


#####miceData

data(miceData)

db<-miceData

db$left_rec<-db$l

db$right_rec<-ifelse(db$u=='Inf', NA, db$u)

db$right_1e9<-ifelse(is.na(db$right_rec), 1e9, db$right_rec)

# Midpoint imputation
library(dplyr)

inputted.data <- db %>%
  mutate(rec.time = ifelse(is.finite(right_rec), (left_rec + right_rec)/2, left_rec),
         status = ifelse(is.finite(right_rec), 1, 0))

head(inputted.data)  #"rec.time", "status" --- two news column -- midpoint

# Make data.frame
results <- data.frame(
  left = db$l,
  right = db$u,
  right_rec = db$right_rec,
  midpoint=inputted.data$rec.time,
  status=inputted.data$status,
  group=inputted.data$grp,
  right_1e9=db$right_1e9
)

# Interval-censored times: model-based imputation + scaled linear redistribution
# Right-censored times (upper bound = Inf): imputed as lower bound to avoid extrapolation

db.ini<-db

db<-db[!is.na(db$right_rec),]

results.ini<-results

results<-results[!is.na(results$right_rec),]


#xgboost_scaled

db$group<-as.numeric(db$grp)

n_bs <- 1000

xgboost.impt<-NULL

for (k in 1:n_bs) {

  cat(k, '\n')
  
  label_lower <- db$left
  label_upper <- results$right_1e9
  
  times<-rep(0,nrow(results)) 
  
  for(i in 1:nrow(results)){
    
    times[i] <- runif(1, min = db$left[i], max = label_upper[i])
  }
  
  X <- model.matrix(~ group, data = results)#[, -1] #-1 tira o coeficiente intercept (retirei estava a dar erro)
  
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

label_lower <- results$left
label_upper <- results$right_1e9

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

#logn_scaled

bs_fits <- vector("list", n_bs)
set.seed(123) 

for (k in 1:n_bs) {
  
  times<-rep(0,nrow(results))
  
  for(i in 1:nrow(results)){
    
    times[i] <- runif(1, min = db$left[i], max = results$right_1e9[i])
  }
  
  results$times<-times
  
  bs_fits[[k]] <- survreg(formula = Surv(times, status) ~group,
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
  
  label_lower <- db$left
  label_upper <- results$right_1e9
  
  val<-label_lower[j]+(times_scaled-min(times_scaled))/(max(times_scaled)-min(times_scaled))*(label_upper[j]-label_lower[j])
  
  val>=results$left[j]
  
  val<results$right[j]
  
  medians[j]<-median(val, na.rm = T)
  
  
}

results$lnorm_1e9_scaled<-medians


#WEIBULL_scaled

bs_fits <- vector("list", n_bs)
set.seed(123)  

for (k in 1:n_bs) {
  
  times<-rep(0,nrow(results))
  
  for(i in 1:nrow(results)){
    
    times[i] <- runif(1, min = db$left[i], max = results$right_1e9[i])
  }
  
  results$times<-times
  
  bs_fits[[k]] <- survreg(formula = Surv(times, status) ~  group,
                          data =  results,
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
  
  label_lower <- db$left
  label_upper <- results$right_1e9
  
  val<-label_lower[j]+(times_scaled-min(times_scaled))/(max(times_scaled)-min(times_scaled))*(label_upper[j]-label_lower[j])
  
  val>=results$left[j]
  
  val<results$right[j]
  
  medians[j]<-median(val, na.rm = T)
  
}

results$wei_1e9_scaled<-medians

#exponential_scaled 

bs_fits <- vector("list", n_bs)
set.seed(123) 

for (k in 1:n_bs) {
  
  times<-rep(0,nrow(results))
  
  for(i in 1:nrow(results)){
    
    times[i] <- runif(1, min = db$left[i], max = results$right_1e9[i])
  }
  
  results$times<-times
  
  bs_fits[[k]] <- survreg(formula = Surv(times, status) ~  group,
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
  
  label_lower <- db$left
  label_upper <- results$right_1e9
  
  val<-label_lower[j]+(times_scaled-min(times_scaled))/(max(times_scaled)-min(times_scaled))*(label_upper[j]-label_lower[j])
  
  val>=results$left[j]
  
  val<results$right[j]
  
  medians[j]<-median(val, na.rm = T)
  
}

results$exp_1e9_scaled<-medians

#Gaussian_scaled 

bs_fits <- vector("list", n_bs)
set.seed(123) 

for (k in 1:n_bs) {
  
  times<-rep(0,nrow(results))
  
  for(i in 1:nrow(results)){
    
    times[i] <- runif(1, min = db$left[i], max = results$right_1e9[i])
  }
  
  results$times<-times
  
  bs_fits[[k]] <- survreg(formula = Surv(times, status) ~  group,
                          data =  results,
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
  
  label_lower <- db$left
  label_upper <- results$right_1e9
  
  val<-label_lower[j]+(times_scaled-min(times_scaled))/(max(times_scaled)-min(times_scaled))*(label_upper[j]-label_lower[j])
  
  val>=results$left[j]
  
  val<results$right[j]
  
  medians[j]<-median(val, na.rm = T)
  
  
}

results$gaus_1e9_scaled<-medians

results.na<-results.ini[is.na(results.ini$right_rec), ]

results.na$xgb_1e9_scaled<-results.na$left     

results.na$lnorm_1e9_scaled<-results.na$left

results.na$wei_1e9_scaled<-results.na$left

results.na$exp_1e9_scaled<-results.na$left 

results.na$gaus_1e9_scaled<-results.na$left

results<-results[, -9]

results<-rbind(results.na, results)

dim(results)

results$status_t<-rep(1, nrow(results))

leftTB<-results$left

rightTB<-results$right

fit.int<-icfit(Surv(leftTB, rightTB, type="interval2") ~ 1)

fit.pm<- survfit(Surv(midpoint, status) ~ 1, data = results)  #status_t

fit.xgh<- survfit(Surv(xgb_1e9_scaled, status) ~ 1, data = results)

fit.lnorm<- survfit(Surv(lnorm_1e9_scaled, status) ~ 1, data = results)

fit.wei<- survfit(Surv(wei_1e9_scaled, status) ~ 1, data = results)

fit.exp<- survfit(Surv(exp_1e9_scaled, status) ~ 1, data = results)

fit.gaus<- survfit(Surv(gaus_1e9_scaled, status) ~ 1, data = results)

#uniformizar PM
dados <- data.frame(time = fit.pm$time,
                    surv = fit.pm$surv)
# --- 1. Criar sequência de tempos uniformes ---
tempos_uniformes <- seq(0, max(dados$time), by = 0.5)
# --- 2. Para cada tempo da sequência, encontrar o valor de sobrevivência mais próximo ---
library(dplyr)
dados_uniformes <- data.frame(time = tempos_uniformes) %>%
  rowwise() %>%
  mutate(surv = dados$surv[which.min(abs(dados$time - time))]) %>%
  ungroup()
# --- 3. Arredondar e visualizar ---
dados_uniformes <- dados_uniformes %>%
  mutate(time = round(time, 1),
         surv = round(surv, 3))
# --- 4. Mostrar primeiras linhas ---
dados_uniformes_PM<-dados_uniformes
tab_PM<-as.data.frame(dados_uniformes_PM)

#uniformizar xgb

dados <- data.frame(time = fit.xgh$time,
                    surv = fit.xgh$surv)
# --- 1. Criar sequência de tempos uniformes ---
tempos_uniformes <- seq(0, max(fit.pm$time), by = 0.5)

# --- 2. Para cada tempo da sequência, encontrar o valor de sobrevivência mais próximo ---
dados_uniformes <- data.frame(time = tempos_uniformes) %>%
  rowwise() %>%
  mutate(surv = dados$surv[which.min(abs(dados$time - time))]) %>%
  ungroup()
# --- 3. Arredondar e visualizar ---
dados_uniformes <- dados_uniformes %>%
  mutate(time = round(time, 1),
         surv = round(surv, 3))
# --- 4. Mostrar primeiras linhas ---
dados_uniformes_XGBoost<-dados_uniformes
tab_XGBoost<-as.data.frame(dados_uniformes_XGBoost)

#uniformizar lnorm
dados <- data.frame(time = fit.lnorm$time,
                    surv = fit.lnorm$surv)
# --- 1. Criar sequência de tempos uniformes ---
tempos_uniformes <- seq(0, max(dados$time), by = 0.5)
# --- 2. Para cada tempo da sequência, encontrar o valor de sobrevivência mais próximo ---

dados_uniformes <- data.frame(time = tempos_uniformes) %>%
  rowwise() %>%
  mutate(surv = dados$surv[which.min(abs(dados$time - time))]) %>%
  ungroup()
# --- 3. Arredondar e visualizar ---
dados_uniformes <- dados_uniformes %>%
  mutate(time = round(time, 1),
         surv = round(surv, 3))
# --- 4. Mostrar primeiras linhas ---
dados_uniformes_lnorm<-dados_uniformes
tab_lnorm<-as.data.frame(dados_uniformes_lnorm)

#uniformizar WEIBULL
dados <- data.frame(time = fit.wei$time,
                    surv = fit.wei$surv)
# --- 1. Criar sequência de tempos uniformes ---
tempos_uniformes <- seq(0, max(dados$time), by = 0.5)
# --- 2. Para cada tempo da sequência, encontrar o valor de sobrevivência mais próximo ---
dados_uniformes <- data.frame(time = tempos_uniformes) %>%
  rowwise() %>%
  mutate(surv = dados$surv[which.min(abs(dados$time - time))]) %>%
  ungroup()
# --- 3. Arredondar e visualizar ---
dados_uniformes <- dados_uniformes %>%
  mutate(time = round(time, 1),
         surv = round(surv, 3))
# --- 4. Mostrar primeiras linhas ---
dados_uniformes_wei<-dados_uniformes
tab_wei<-as.data.frame(dados_uniformes_wei)

#uniformizar EXP
dados <- data.frame(time = fit.exp$time,
                    surv = fit.exp$surv)
# --- 1. Criar sequência de tempos uniformes ---
tempos_uniformes <- seq(0, max(dados$time), by = 0.5)
# --- 2. Para cada tempo da sequência, encontrar o valor de sobrevivência mais próximo ---
dados_uniformes <- data.frame(time = tempos_uniformes) %>%
  rowwise() %>%
  mutate(surv = dados$surv[which.min(abs(dados$time - time))]) %>%
  ungroup()
# --- 3. Arredondar e visualizar ---
dados_uniformes <- dados_uniformes %>%
  mutate(time = round(time, 1),
         surv = round(surv, 3))
# --- 4. Mostrar primeiras linhas ---
dados_uniformes_exp<-dados_uniformes
tab_exp<-as.data.frame(dados_uniformes_exp)

#uniformizar GAUSS
dados <- data.frame(time = fit.gaus$time,
                    surv = fit.gaus$surv)
# --- 1. Criar sequência de tempos uniformes ---
tempos_uniformes <- seq(0, max(dados$time), by = 0.5)
# --- 2. Para cada tempo da sequência, encontrar o valor de sobrevivência mais próximo ---
dados_uniformes <- data.frame(time = tempos_uniformes) %>%
  rowwise() %>%
  mutate(surv = dados$surv[which.min(abs(dados$time - time))]) %>%
  ungroup()
# --- 3. Arredondar e visualizar ---
dados_uniformes <- dados_uniformes %>%
  mutate(time = round(time, 1),
         surv = round(surv, 3))
# --- 4. Mostrar primeiras linhas ---
dados_uniformes_gaus<-dados_uniformes
tab_gaus<-as.data.frame(dados_uniformes_gaus)

tab_PM<-tab_PM[1:3500,]
tab_XGBoost<-tab_XGBoost[1:3500,]
tab_lnorm<-tab_lnorm[1:3500,]
tab_wei<-tab_wei[1:3500,]
tab_exp<-tab_exp[1:3500,]
tab_gaus<-tab_gaus[1:3500,]

li<-c(fit.int$intmap[1,] )
sur<-c(1-cumsum(fit.int$pf))

cbind(li, sur)

#PLOT (FIG.1)

plot(fit.int, ylim=c(0,1)) #, xlim=c(0,1050)
lines(tab_PM[1:2000,1], tab_PM[1:2000,2],  col=2,lty = 1, lwd = 1.2)
lines(tab_XGBoost[1:2000,1], tab_XGBoost[1:2000,2],  col=3,lty = 1, lwd = 1.2)
lines(tab_lnorm[1:2000,1], tab_lnorm[1:2000,2],  col=4,lty = 2, lwd = 1.2)

lines(tab_wei[1:2800,1], tab_wei[1:2800,2],  col=2,lty = 2, lwd = 1.2)
lines(tab_exp[1:2800,1], tab_exp[1:2800,2],  col=3,lty = 2, lwd = 1.2)
lines(tab_gaus[1:2800,1], tab_gaus[1:2800,2],  col=4,lty = 2, lwd = 1.2)
legend("topright", legend = c("TB", "Midpoint", "XGBoost", "Log-normal", "Weibull",
                              "Exponential", "Gauss"), lwd = 1.2, 
       col = c(1,2,3,4, 2, 3, 4), cex = 0.5, 
       lty = c(1,1,1,1, 2, 2,2))