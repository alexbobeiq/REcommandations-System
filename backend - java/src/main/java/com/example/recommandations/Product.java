package com.example.recommandations;

import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import lombok.Data;
import lombok.Getter;

@Data
@Entity
@Table(name="Products")
public class Product {

    @Id
    private String code;
    private String name;
    private Double price;


    public Product() {}


    public Product(String code, String name, Double price) {
        this.code = code;
        this.name = name;
        this.price = price;
    }


    public void setCode(String code) {
        this.code = code;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setPrice(Double price) {
        this.price = price;
    }

    public String getCode() {
        return code;
    }

    public String getName() {
        return name;
    }

    public Double getPrice() {
        return price;
    }
}