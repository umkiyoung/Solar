# 태양전지연구실 AI프로젝트

## Weighted_Average_Reflectance_of_Solar_Cells

- prototype_1
    : 기존 데이터 이용
    - RGB -> 두께
    - RGB + alpha -> Reflectance 
- prototype_2
    : 새로운 데이터 이용
    - RGB -> 두께 (완), 결과 양호
    - RGB + alpha -> Reflectance (진행중)
-----
## Prediction of Contact Resistivity

- Contact_Resistivity.ipynb 파일 참조
----
## Optimal Experimental Date and Efficiency Prediction of Perovskite Solar Cells based on Weather Forecast

- dewpoint_crawling.ipynb 파일 참조
-----

## SEM image -> 반사도 예측

- Sample_test.ipynb 파일 참조
    - 다른 image 타입은 시도 x texture만 진행
    - plot 9, 10에서 성능 저하 확인
    - weighted reflectance 처럼 y 값이 하나인 데이터가 있어야 이미지 데이터의 영향력 확인 가능.
